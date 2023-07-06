#include "transfusion/transfusion.hpp"

#include <open3d/Open3D.h>
#include <open3d/core/TensorFunction.h>

#include <fstream>
#include <open3d/visualization/visualizer/RenderOption.h>

namespace {
bool read_pcd_bin(const std::string &fname,
                  open3d::t::geometry::PointCloud &cloud) {
  if (fname.empty())
    return false;
  std::ifstream file(fname, std::ios::in | std::ios::binary);
  if (!file.good()) {
    std::cerr << "Error during openning the file: " << fname << std::endl;
    return false;
  }

  open3d::core::Dtype dtype_f{open3d::core::Dtype::Float32};
  open3d::core::Device device_type{open3d::core::Device::DeviceType::CPU, 0};
  std::vector<float> intensities_buffer;
  std::vector<Eigen::Vector3d> points_buffer;
  float x, y, z, i, ring;
  while (file.read(reinterpret_cast<char *>(&x), sizeof(float)) &&
         file.read(reinterpret_cast<char *>(&y), sizeof(float)) &&
         file.read(reinterpret_cast<char *>(&z), sizeof(float)) &&
         file.read(reinterpret_cast<char *>(&i), sizeof(float)) &&
         file.read(reinterpret_cast<char *>(&ring), sizeof(float))) {
    points_buffer.push_back(Eigen::Vector3d(x, y, z));
    intensities_buffer.push_back(i);
  }
  std::cout << "base points num: " << points_buffer.size() << std::endl;
  std::vector<float> intensities_buffer_not_close;
  std::vector<Eigen::Vector3d> points_buffer_not_close;
  intensities_buffer_not_close.reserve(points_buffer.size());
  points_buffer_not_close.reserve(points_buffer.size());
  for (size_t i = 0; i < points_buffer.size(); ++i) {
    if (std::abs(points_buffer.at(i)[0]) < 1.0 &&
        std::abs(points_buffer.at(i)[1]) < 1.0) {
      continue;
    }
    intensities_buffer_not_close.push_back(intensities_buffer.at(i));
    points_buffer_not_close.push_back(points_buffer.at(i));
  }
  std::cout << "not close points num: " << points_buffer_not_close.size()
            << std::endl;

  for (size_t i = 0; i < 9; ++i) {
    intensities_buffer.insert(intensities_buffer.end(),
                              intensities_buffer_not_close.begin(),
                              intensities_buffer_not_close.end());
    points_buffer.insert(points_buffer.end(), points_buffer_not_close.begin(),
                         points_buffer_not_close.end());
  }
  std::cout << "sweep points num: " << points_buffer.size() << std::endl;
  std::vector<float> times_buffer(points_buffer.size(), 0.0f);
  open3d::core::Tensor positions =
      open3d::core::eigen_converter::EigenVector3dVectorToTensor(
          points_buffer, dtype_f, device_type);
  cloud.SetPointPositions(positions);
  auto intensity = open3d::core::Tensor(
      intensities_buffer, {1, static_cast<int>(intensities_buffer.size())},
      open3d::core::Dtype::Float32);
  auto time = open3d::core::Tensor(times_buffer,
                                   {1, static_cast<int>(times_buffer.size())},
                                   open3d::core::Dtype::Float32);
  cloud.SetPointAttr("intensity", intensity);
  cloud.SetPointAttr("time", time);
  file.close();
  return true;
}
} // namespace

int main(int argc, char **argv) {
  assert(argc == 3);
  auto cloud_ptr = std::make_shared<open3d::t::geometry::PointCloud>();
  if (read_pcd_bin(std::string(argv[1]), *cloud_ptr)) {
    open3d::utility::LogInfo("Successfully read {}", argv[1]);
  } else {
    open3d::utility::LogInfo("Failed to read {}", argv[1]);
    return -1;
  }
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto transfusion_impl =
      std::make_unique<transfusion::Transfusion>(std::string(argv[2]), stream);
  using Color = Eigen::Vector3d;
  std::vector<Color> color_map = {
      Color(255, 158, 0) / 255.0,  // Orange
      Color(255, 99, 71) / 255.0,  // Tomato
      Color(255, 140, 0) / 255.0,  // Darkorange
      Color(255, 127, 80) / 255.0, // Coral
      Color(233, 150, 70) / 255.0, // Darksalmon
      Color(220, 20, 60) / 255.0,  // Crimson
      Color(255, 61, 99) / 255.0,  // Red
      Color(0, 0, 230) / 255.0,    // Blue
      Color(47, 79, 79) / 255.0,   // Darkslategrey
      Color(112, 128, 144) / 255.0 // Slategrey
  };

  auto positions = cloud_ptr->GetPointPositions();
  auto intensities = cloud_ptr->GetPointAttr("intensity").Reshape({-1, 1});
  auto times = cloud_ptr->GetPointAttr("time").Reshape({-1, 1});
  auto concat = open3d::core::Concatenate({positions, intensities, times}, 1);
  float *d_points = nullptr;
  int num_points = cloud_ptr->ToLegacy().points_.size();
  float *tmp = (float *)concat.GetDataPtr();
  checkCudaErrors(
      cudaMalloc((void **)&d_points, num_points * 5 * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_points, concat.GetDataPtr(),
                             num_points * 5 * sizeof(float),
                             cudaMemcpyHostToDevice));
  std::vector<transfusion::Bndbox> boxes;
  transfusion_impl->infer((void *)d_points, num_points, boxes);
  open3d::visualization::VisualizerWithKeyCallback vis;
  vis.CreateVisualizerWindow("Point Cloud Viewer", 1600, 900);
  vis.GetRenderOption().background_color_ = {0.0, 0.0, 0.0};
  vis.GetRenderOption().point_color_option_ =
      open3d::visualization::RenderOption::PointColorOption::Color;
  vis.GetRenderOption().point_size_ = 1.0;
  vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(
      cloud_ptr->ToLegacy().PaintUniformColor({1.0, 1.0, 1.0})));
  for (const auto &box : boxes) {
    if (box.score < 0.2) {
      continue;
    }
    if (box.x < -61.2 || 61.2 < box.x || box.y < -61.2 || 61.2 < box.y ||
        box.z < -10.0 || 10.0 < box.z) {
      continue;
    }
    Eigen::Matrix3d R;
    R << std::cos(box.rt), -std::sin(box.rt), 0.0, std::sin(box.rt),
        std::cos(box.rt), 0.0, 0.0, 0.0, 1.0;
    auto bbox_ptr = std::make_shared<open3d::geometry::OrientedBoundingBox>(
        Eigen::Vector3d{box.x, box.y, box.z}, R,
        Eigen::Vector3d{box.w, box.l, box.h});
    bbox_ptr->color_ = color_map.at(box.id);
    vis.AddGeometry(bbox_ptr);
  }

  vis.Run();
  vis.DestroyVisualizerWindow();
  cudaStreamDestroy(stream);

  return 0;
}

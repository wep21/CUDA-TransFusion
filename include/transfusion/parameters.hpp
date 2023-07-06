
namespace transfusion {
struct Parameters {
  int num_classes = 10;
  float min_x_range = -51.2;
  float max_x_range = 51.2;
  float min_y_range = -51.2;
  float max_y_range = 51.2;
  float min_z_range = -5.0;
  float max_z_range = 3.0;
  // the size of a pillar
  float pillar_x_size = 0.2;
  float pillar_y_size = 0.2;
  float pillar_z_size = 8.0;
  int out_size_factor = 4;
  int max_num_points_per_pillar = 20;
  int num_point_values = 4;
  int num_proposals = 200;
  // the number of feature maps for pillar scatter
  int num_feature_scatter = 64;
  // the score threshold for classification
  float score_thresh = 0.2;
  float nms_thresh = 0.01;
  int max_num_pillars = 40000;
  int pillar_points_bev = max_num_points_per_pillar * max_num_pillars;
  // the detected boxes result decode by (x, y, z, w, l, h, yaw)
  int num_box_values = 8;
  // the input size of the 2D backbone network
  int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;
  int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;
  int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;
  // the output size of the 2D backbone network
  int feature_x_size = grid_x_size / 4;
  int feature_y_size = grid_y_size / 4;
  Parameters(){};
};
} // namespace transfusion
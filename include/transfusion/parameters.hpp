
namespace transfusion {
struct Parameters {
  int num_classes = 5;
  float min_x_range = -102.4;
  float max_x_range = 102.4;
  float min_y_range = -102.4;
  float max_y_range = 102.4;
  float min_z_range = -4.0;
  float max_z_range = 6.0;
  // the size of a pillar
  float pillar_x_size = 0.32;
  float pillar_y_size = 0.32;
  float pillar_z_size = 10.0;
  int out_size_factor = 1;
  int max_num_points_per_pillar = 20;
  int num_point_values = 4;
  int num_proposals = 500;
  // the number of feature maps for pillar scatter
  int num_feature_scatter = 64;
  // the score threshold for classification
  float score_thresh = 0.2;
  float nms_thresh = 0.01;
  int max_num_pillars = 60000;
  int pillar_points_bev = max_num_points_per_pillar * max_num_pillars;
  // the detected boxes result decode by (x, y, z, w, l, h, yaw)
  int num_box_values = 8;
  // the input size of the 2D backbone network
  int grid_x_size = (max_x_range - min_x_range) / pillar_x_size;
  int grid_y_size = (max_y_range - min_y_range) / pillar_y_size;
  int grid_z_size = (max_z_range - min_z_range) / pillar_z_size;
  // the output size of the 2D backbone network
  int feature_x_size = grid_x_size / out_size_factor;
  int feature_y_size = grid_y_size / out_size_factor;
  Parameters(){};
};
} // namespace transfusion
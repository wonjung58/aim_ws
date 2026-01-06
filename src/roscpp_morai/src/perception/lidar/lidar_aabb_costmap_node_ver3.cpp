// lidar_aabb_costmap_node.cpp (modified: TF 안정화 + frame/stamp 강제 통일)
// - PointCloud를 costmap_frame(기본 base_link)로 변환(필수)
// - costmap / marker / debug cloud 모두 frame_id를 costmap_frame으로 강제
// - TF lookup은 시뮬에서 extrapolation 줄이려고 ros::Time(0) (latest) 사용

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <algorithm>
#include <memory>

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>

// =================refactoring==================

// Config
#include "config/lidar_aabb_params.hpp"

using std::string;

// Global params (loaded from config)
static LidarAabbParams g_params;

// ------------------------------
// Publishers
// ------------------------------
ros::Publisher pub_costmap;
ros::Publisher pub_ransac;
ros::Publisher pub_cluster_all;
ros::Publisher pub_cluster_cav;
ros::Publisher pub_cav_aabb_centers;
ros::Publisher pub_cav_aabb_boxes;
ros::Publisher pub_cav_aabb_center_spheres;

// ------------------------------
// TF
// ------------------------------
std::unique_ptr<tf2_ros::Buffer> tfBufferPtr;
std::unique_ptr<tf2_ros::TransformListener> tfListenerPtr;

// Helper function for frame name normalization
static inline std::string normalizeFrame(std::string f) {
  while (!f.empty() && f.front() == '/') f.erase(f.begin());
  return f;
}

// Detection struct
struct Detection {
  int id = -1;
  ros::Time stamp;

  Eigen::Vector3f centroid;
  Eigen::Vector3f min_pt;
  Eigen::Vector3f max_pt;
  Eigen::Vector3f size;

  int num_points = 0;
  float range = 0.0f;
};

// init costmap (must be before visualize which uses worldToGrid)
#include "costmap/costmap_init.hpp"

// visualize
#include "visualize/lidar_aabb_visualize.hpp"

// pointcloud processing
#include "pointcloud_processing/pointcloud_processing.hpp"

// callback
#include "lidar_callback/lidar_callback.hpp"
// ===============================================


// ------------------------------
// main
// ------------------------------
int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar_aabb_costmap_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // Load all parameters from config
  loadParams(pnh, g_params);
  g_params.costmap_frame = normalizeFrame(g_params.costmap_frame);

  // TF init (cache time 늘려서 시뮬에서 lookup 여유)
  tfBufferPtr.reset(new tf2_ros::Buffer(ros::Duration(10.0)));
  tfListenerPtr.reset(new tf2_ros::TransformListener(*tfBufferPtr));

  // publishers
  pub_costmap = nh.advertise<nav_msgs::OccupancyGrid>(g_params.costmap_topic, 1);
  pub_ransac = nh.advertise<sensor_msgs::PointCloud2>("/ransac", 1);
  pub_cluster_all = nh.advertise<sensor_msgs::PointCloud2>("/cluster", 1);
  pub_cluster_cav = nh.advertise<sensor_msgs::PointCloud2>("/cluster_cav", 1);

  pub_cav_aabb_centers = nh.advertise<geometry_msgs::PoseArray>("/cav_aabb_centers", 1);
  pub_cav_aabb_boxes = nh.advertise<visualization_msgs::MarkerArray>("/cav_aabb_boxes", 1);
  pub_cav_aabb_center_spheres = nh.advertise<visualization_msgs::MarkerArray>("/cav_aabb_center_spheres", 1);

  // subscriber
  ros::Subscriber sub = nh.subscribe(g_params.lidar_topic, 1, lidarCallback);

  printLidarAABB_statistics(g_params);
  ros::spin();
  return 0;
}



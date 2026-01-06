#ifndef GLOBAL_HPP
#define GLOBAL_HPP

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

// TF2
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>

using namespace std;

// ========================= Type ========================= //
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef uint64_t uint64;
typedef short int16;
typedef int int32;
typedef float float32;
typedef double float64;
typedef char char8;

// ========================= Struct ========================= //
// struct Publisher
// {
//     ros::Publisher pub_costmap;
//     ros::Publisher pub_ransac;
//     ros::Publisher pub_cluster_all;
//     ros::Publisher pub_cluster_cav;
//     ros::Publisher pub_cav_aabb_centers;
//     ros::Publisher pub_cav_aabb_boxes;
//     ros::Publisher pub_cav_aabb_center_spheres;
// };

// struct Subscriber
// {
//     ros::Subscriber sub_lidar;
// };

struct TF
{
    unique_ptr<tf2_ros::Buffer> tfBufferPtr;
    unique_ptr<tf2_ros::TransformListener> tfListenerPtr;
};

struct LidarParam
{
    string lidar_topic;
    string costmap_topic;

    string frame_id;
    string costmap_frame;

    // costmap params
    float map_resolution;
    float map_width;
    float map_height;
    int8_t unknown_cost;
    int8_t free_cost;
    int8_t obstacle_cost;
    float inflation_m;

    // lidar prefilter params
    float min_height;
    float max_height;
    float lidar_range;
    float voxel_leaf;


    // ego vehicle ROI params
    float ego_exclusion; // car_size 로 변경 고려

    // ransac params
    float ransac_dist_thresh;
    float ransac_eps_angle_deg;
    int ransac_max_iter;

    // clustering params
    float cluster_tolerance;
    int cluster_min_size;
    int cluster_max_size;

    // vehicle 후보 조건
    int cav_min_points;
    float cav_dx_min;
    float cav_dx_max;
    float cav_dy_min;
    float cav_dy_max;
    float cav_dz_min;
    float cav_dz_max;

    bool use_tf;
    bool require_tf;
    bool publish_cluster_all;
    float car_size;

    // cost 값 (박스 컬러별)
    int8_t cav_cost_green;
    int8_t cav_cost_red;

    // visualization params
    float box_alpha;
    float marker_lifetime;
    float center_sphere_diam;
};

struct LidarCluster
{

};

struct Lidar
{
    sensor_msgs::PointCloud2ConstPtr input_cloud;

    pcl::PointCloud <pcl::PointXYZI>::Ptr raw_cloud;
    pcl::PointCloud <pcl::PointXYZI>::Ptr roi_cloud;
    pcl::PointCloud <pcl::PointXYZI>::Ptr voxel_cloud;
    pcl::PointCloud <pcl::PointXYZI>::Ptr ransac_cloud;
    std::vector<pcl::PointIndices> cluster_indices;

    // LidarParam param;
    // LidarCluster cluster;
    Lidar()
    {
        raw_cloud.reset(new pcl::PointCloud <pcl::PointXYZI>);
        roi_cloud.reset(new pcl::PointCloud <pcl::PointXYZI>);
        voxel_cloud.reset(new pcl::PointCloud <pcl::PointXYZI>);
        ransac_cloud.reset(new pcl::PointCloud <pcl::PointXYZI>);
    }
};

#endif
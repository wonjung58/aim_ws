#include <Global/Global.hpp>

// #include <ros/ros.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <nav_msgs/OccupancyGrid.h>
// #include <geometry_msgs/PoseArray.h>
// #include <visualization_msgs/MarkerArray.h>
// #include <visualization_msgs/Marker.h>

// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>

// #include <pcl/filters/passthrough.h>
// #include <pcl/filters/voxel_grid.h>

// #include <pcl/ModelCoefficients.h>
// #include <pcl/sample_consensus/method_types.h>
// #include <pcl/sample_consensus/model_types.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/filters/extract_indices.h>

// #include <pcl/search/kdtree.h>
// #include <pcl/segmentation/extract_clusters.h>

// #include <Eigen/Dense>
// #include <limits>
// #include <cmath>
// #include <algorithm>

// // TF2
// #include <tf2_ros/transform_listener.h>
// #include <tf2_sensor_msgs/tf2_sensor_msgs.h>
// #include <geometry_msgs/TransformStamped.h>

using namespace std;

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

// ------------------------------
// Params
// ------------------------------
string lidar_topic = "/lidar3D";
string costmap_topic = "/costmap_from_lidar";

bool use_tf = false;                 // TF로 cloud를 costmap_frame으로 변환할지
bool require_tf = false;             // TF 실패하면 프레임 스킵할지
string costmap_frame = "lidar_link"; // costmap/marker publish frame

// costmap params
float map_resolution = 0.05f;  // 5cm
float map_width = 50.0f;       // meters
float map_height = 50.0f;      // meters
int8_t unknown_cost = -1;      // OccupancyGrid: unknown=-1 권장
int8_t free_cost = 0;          // free
int8_t obstacle_cost = 100;    // occupied

float inflation_m = 0.0f;      // AABB를 주변으로 더 두껍게 (m)

// lidar prefilter
float min_height = -0.5f;
float max_height = 0.5f;
float lidar_range = 25.0f;
float voxel_leaf = 0.10f;

// ego 제거 ROI
float ego_exclusion = 2.0f; // |x|<=2 and |y|<=2 영역 제거

// RANSAC
float ransac_dist_thresh = 0.30f;
float ransac_eps_angle_deg = 10.0f;
int ransac_max_iter = 200;

// clustering
float cluster_tolerance = 0.4f;
int cluster_min_size = 20;
int cluster_max_size = 2000;

// vehicle 후보 조건
int cav_min_points = 10;
float cav_dx_min = 0.3f, cav_dx_max = 8.0f;
float cav_dy_min = 0.2f, cav_dy_max = 4.0f;
float cav_dz_min = 0.1f, cav_dz_max = 3.0f;

// cost 값 (박스 컬러별)
int8_t cav_cost_green = 75;
int8_t cav_cost_red   = 100;

// marker
float box_alpha = 0.25f;
float marker_lifetime = 0.2f;
float center_sphere_diam = 0.3f;

// ------------------------------
// Detection struct
// ------------------------------
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

// ------------------------------
// Helpers: costmap indexing
// ------------------------------

// grid index가 맵 밖으로 나갔을 때 경계 안으로 잘라 넣는 용도
static inline int clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

// costmap 메시지 초기화
static inline void initCostmap(nav_msgs::OccupancyGrid &cm,
                               const std_msgs::Header &hdr)
{
  cm.header = hdr;
  cm.info.resolution = map_resolution; // 한 셀 크기 (m)

  // 맵 크기 (총 셀 개수)
  const int w = (int)std::round(map_width / map_resolution);
  const int h = (int)std::round(map_height / map_resolution);

  cm.info.width = w;
  cm.info.height = h;

  // 로컬 costmap: costmap 원점이 프레임(ego) 기준 (0,0)을 중앙에 두도록
  // origin.position : 맵의 왼쪽 아래 코너 위치 = costmap 원점
  cm.info.origin.position.x = -map_width * 0.5;
  cm.info.origin.position.y = -map_height * 0.5;
  cm.info.origin.position.z = 0.0;
  cm.info.origin.orientation.w = 1.0;

  // 모든 셀 unknown으로 초기화
  cm.data.assign(w * h, unknown_cost);
}

// ------------------------------
// 월드 좌표 -> 그리드 좌표 변환
// 월드 좌표가 맵 안에 있는지 검사
// ------------------------------
static inline bool worldToGrid(const nav_msgs::OccupancyGrid &cm,
                               float x, float y,
                               int &gx, int &gy)
{
  // 월드 좌표 (x, y) -> 그리드 좌표 (gx, gy)
  gx = (int)std::floor((x - cm.info.origin.position.x) / cm.info.resolution);
  gy = (int)std::floor((y - cm.info.origin.position.y) / cm.info.resolution);

  if (gx < 0 || gx >= (int)cm.info.width) return false;
  if (gy < 0 || gy >= (int)cm.info.height) return false;
  return true;
}

static inline int gridIndex(const nav_msgs::OccupancyGrid &cm, int gx, int gy)
{
  if (gx < 0 || gx >= (int)cm.info.width) return -1;
  if (gy < 0 || gy >= (int)cm.info.height) return -1;
  return gy * (int)cm.info.width + gx;
}

static void paintAABB(nav_msgs::OccupancyGrid &cm,
                      float min_x, float min_y, float max_x, float max_y,
                      int8_t value, float inflation = 0.0f)
{
  // min/max 정렬
  if (min_x > max_x) swap(min_x, max_x);
  if (min_y > max_y) swap(min_y, max_y);

  // inflation 적용
  min_x -= inflation; min_y -= inflation;
  max_x += inflation; max_y += inflation;

  int gx0, gy0, gx1, gy1;
  // AABB 모서리를 grid 좌표로 변환
  if (!worldToGrid(cm, min_x, min_y, gx0, gy0)) {
    // clamp를 위해 일단 계산만
    gx0 = (int)floor((min_x - cm.info.origin.position.x) / cm.info.resolution);
    gy0 = (int)floor((min_y - cm.info.origin.position.y) / cm.info.resolution);
  }
  if (!worldToGrid(cm, max_x, max_y, gx1, gy1)) {
    gx1 = (int) floor((max_x - cm.info.origin.position.x) / cm.info.resolution);
    gy1 = (int) floor((max_y - cm.info.origin.position.y) / cm.info.resolution);
  }

  if (gx0 > gx1) swap(gx0, gx1);
  if (gy0 > gy1) swap(gy0, gy1);

  gx0 = clampi(gx0, 0, (int)cm.info.width  - 1);
  gx1 = clampi(gx1, 0, (int)cm.info.width  - 1);
  gy0 = clampi(gy0, 0, (int)cm.info.height - 1);
  gy1 = clampi(gy1, 0, (int)cm.info.height - 1);

  const int w = (int)cm.info.width;
  for (int y = gy0; y <= gy1; ++y) {
    for (int x = gx0; x <= gx1; ++x) {
      int idx = y * w + x;
      if (idx >= 0 && idx < (int)cm.data.size()) {
        // 최대값으로 덮기 (이미 장애물이면 유지)
        cm.data[idx] = std::max(cm.data[idx], value);
      }
    }
  }
}

// ------------------------------
// Helpers: point cloud processing
// ------------------------------
static void filterByHeight(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                           float zmin, float zmax)
{
  pcl::PassThrough<pcl::PointXYZI> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(zmin, zmax);
  pass.filter(*cloud);
}

static void filterByRange(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                          float range_max)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr out(new pcl::PointCloud<pcl::PointXYZI>);
  out->reserve(cloud->size());
  for (auto &p : cloud->points) {
    float r = std::sqrt(p.x*p.x + p.y*p.y);
    if (r <= range_max) out->push_back(p);
  }
  out->width = (uint32_t)out->size();
  out->height = 1;
  cloud.swap(out);
}

static void voxelDownsample(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, float leaf)
{
  pcl::VoxelGrid<pcl::PointXYZI> voxel;
  voxel.setInputCloud(cloud);
  voxel.setLeafSize(leaf, leaf, leaf);
  voxel.filter(*cloud);
}

static pcl::PointCloud<pcl::PointXYZI>::Ptr removeEgoROI(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, float s)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr out(new pcl::PointCloud<pcl::PointXYZI>);
  out->reserve(cloud->size());
  for (auto &p : cloud->points) {
    if (std::fabs(p.x) <= s && std::fabs(p.y) <= s) continue;
    out->push_back(p);
  }
  out->width = (uint32_t)out->size();
  out->height = 1;
  return out;
}

static pcl::PointCloud<pcl::PointXYZI>::Ptr ransacRemoveGround(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
  float dist_thresh, float eps_angle_deg, int max_iter,
  pcl::ModelCoefficients::Ptr coeff_out = nullptr)
{
  // 간단한 높이 기반 필터링: z > 0.1m만 유지 (지면 제거)
  // RANSAC 세그멘테이션 불안정성 제거
  pcl::PointCloud<pcl::PointXYZI>::Ptr out(new pcl::PointCloud<pcl::PointXYZI>);
  out->reserve(cloud->size());
  
  float ground_threshold = 0.05f;  // 지면 높이: 5cm 이상만 유지
  for (const auto &p : cloud->points) {
    if (p.z > ground_threshold) {
      out->push_back(p);
    }
  }
  
  out->width = (uint32_t)out->size();
  out->height = 1;
  
  if (out->empty()) {
    // 높이 필터 실패 시 원본 반환
    return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>(*cloud));
  }
  
  return out;
}

static std::vector<pcl::PointIndices> euclideanCluster(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
  float tol, int min_sz, int max_sz)
{
  pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
  kdtree->setInputCloud(cloud);

  pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
  ec.setInputCloud(cloud);
  ec.setSearchMethod(kdtree);
  ec.setClusterTolerance(tol);
  ec.setMinClusterSize(min_sz);
  ec.setMaxClusterSize(max_sz);

  std::vector<pcl::PointIndices> indices;
  ec.extract(indices);
  return indices;
}


// ------------------------------
// Helpers: build Detection list
// 각 클러스터에 대해 AABB + 중심 + 크기 등을 계산 -> Detection 리스트 반환
// ------------------------------

static std::vector<Detection> buildDetections(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, // ransac_cloud
  const std::vector<pcl::PointIndices> &clusters,
  const ros::Time &stamp)
{
  std::vector<Detection> dets;
  dets.reserve(clusters.size());

  for (int i = 0; i < (int)clusters.size(); ++i) {
    const auto &idxs = clusters[i].indices;
    if (idxs.empty()) continue;

    // AABB centroid
    double sum_x = 0, sum_y = 0, sum_z = 0; 

    // AABB min/max
    float min_x =  std::numeric_limits<float>::infinity();
    float min_y =  std::numeric_limits<float>::infinity();
    float min_z =  std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();

    int count = 0;
    for (int idx : idxs) {
      if (idx < 0 || idx >= (int)cloud->size()) continue;
      const auto &p = cloud->points[idx];

      sum_x += p.x; sum_y += p.y; sum_z += p.z;

      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      min_z = std::min(min_z, p.z);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
      max_z = std::max(max_z, p.z);
      ++count;
    }
    if (count == 0) continue;

    Detection d;
    d.id = i;
    d.stamp = stamp;
    d.num_points = count;

    const float cx = (float)(sum_x / count);
    const float cy = (float)(sum_y / count);
    const float cz = (float)(sum_z / count);

    d.centroid = Eigen::Vector3f(cx, cy, cz);
    d.min_pt = Eigen::Vector3f(min_x, min_y, min_z);
    d.max_pt = Eigen::Vector3f(max_x, max_y, max_z);
    // AABB 크기
    d.size   = d.max_pt - d.min_pt;
    // XY 평면에서의 거리
    d.range  = std::sqrt(cx*cx + cy*cy);
    dets.push_back(d);
  }

  return dets;
}


// ------------------------------
// Helpers: vehicle 후보 조건 검사
// ------------------------------
static bool isCavCandidate(const Detection &d)
{
  const float dx = d.size.x();
  const float dy = d.size.y();
  const float dz = d.size.z();

  // vehicle cluster의 num_points, 높이, 길이, 폭 조건이므로
  // 원기둥 등의 장애물을 걸러내고 있을 것 -> 원기둥 클러스터 조건 함수 생성해야

  if (d.num_points < cav_min_points) return false;
  if (dz < cav_dz_min || dz > cav_dz_max) return false;
  if (dx < cav_dx_min || dx > cav_dx_max) return false;
  if (dy < cav_dy_min || dy > cav_dy_max) return false;
  return true;
}


// ------------------------------
// Helpers: visualization
// ------------------------------
static void publishColoredClustersAll(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
  const std::vector<pcl::PointIndices> &clusters,
  const std_msgs::Header &hdr)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  colored->points.reserve(cloud->points.size());
  colored->is_dense = true;

  int cluster_id = 0;
  for (const auto &c : clusters) {
    uint8_t r=0,g=0,b=0;
    switch (cluster_id % 6) {
      case 0: r=255; g= 80; b= 80; break;
      case 1: r= 80; g=255; b= 80; break;
      case 2: r= 80; g= 80; b=255; break;
      case 3: r=255; g=255; b= 80; break;
      case 4: r=255; g= 80; b=255; break;
      case 5: r= 80; g=255; b=255; break;
    }

    for (int idx : c.indices) {
      if (idx < 0 || idx >= (int)cloud->size()) continue;
      const auto &src = cloud->points[idx];
      pcl::PointXYZRGB p;
      p.x = src.x; p.y = src.y; p.z = src.z;
      p.r = r; p.g = g; p.b = b;
      colored->points.push_back(p);
    }
    cluster_id++;
  }

  colored->width = (uint32_t)colored->points.size();
  colored->height = 1;

  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*colored, msg);
  msg.header = hdr;
  pub_cluster_all.publish(msg);
}

// ------------------------------
// Main callback
// ------------------------------
void lidarCallback(const sensor_msgs::PointCloud2ConstPtr &in_msg)
{
  if (!ros::ok()) return;

  // 1) TF transform to costmap_frame (optional)
  sensor_msgs::PointCloud2 cloud_tf;
  std_msgs::Header hdr = in_msg->header;

  if (use_tf && hdr.frame_id != costmap_frame) {
    try {
      geometry_msgs::TransformStamped tf =
        tfBufferPtr->lookupTransform(costmap_frame, hdr.frame_id, hdr.stamp, ros::Duration(0.05));

      tf2::doTransform(*in_msg, cloud_tf, tf);
      hdr = cloud_tf.header;           // frame_id = costmap_frame
    } catch (const std::exception &e) {
      ROS_WARN_THROTTLE(1.0, "TF failed (%s -> %s): %s",
                        hdr.frame_id.c_str(), costmap_frame.c_str(), e.what());
      if (require_tf) return;
      cloud_tf = *in_msg; // fallback: 그대로 처리
    }
  } else {
    cloud_tf = *in_msg;
  }

  // 2) ROS -> PCL
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(cloud_tf, *cloud);

  if (cloud->empty()) return;

  // 3) filters
  filterByHeight(cloud, min_height, max_height);
  filterByRange(cloud, lidar_range);

  if (cloud->empty()) return;

  voxelDownsample(cloud, voxel_leaf);

  // 4) remove ego ROI
  pcl::PointCloud<pcl::PointXYZI>::Ptr roi_cloud = removeEgoROI(cloud, ego_exclusion);
  if (roi_cloud->empty()) return;

  // 5) RANSAC ground removal
  pcl::PointCloud<pcl::PointXYZI>::Ptr ransac_cloud =
    ransacRemoveGround(roi_cloud, ransac_dist_thresh, ransac_eps_angle_deg, ransac_max_iter);

  // publish ransac cloud (debug)
  {
    sensor_msgs::PointCloud2 out;
    pcl::toROSMsg(*ransac_cloud, out);
    out.header = hdr;
    if (!out.header.frame_id.empty() && use_tf) out.header.frame_id = hdr.frame_id;
    pub_ransac.publish(out);
  }

  if (ransac_cloud->empty()) return;

  // 6) clustering
  std::vector<pcl::PointIndices> clusters =
    euclideanCluster(ransac_cloud, cluster_tolerance, cluster_min_size, cluster_max_size);

  // publish all colored clusters
  publishColoredClustersAll(ransac_cloud, clusters, hdr);

  // 7) detect clusters info
  std::vector<Detection> dets = buildDetections(ransac_cloud, clusters, hdr.stamp);

  // messages to publish
  geometry_msgs::PoseArray centers_msg;
  centers_msg.header = hdr;

  visualization_msgs::MarkerArray boxes_msg;
  visualization_msgs::MarkerArray center_spheres_msg;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cav_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  cav_colored->is_dense = true;

  int cav_id = 0;

  // 8) init costmap (local around origin)
  nav_msgs::OccupancyGrid costmap;
  // costmap frame을 param으로 강제하고 싶으면 hdr.frame_id를 costmap_frame으로 덮어도 됨
  // 여기서는 hdr.frame_id를 그대로 사용(= TF 변환됐으면 costmap_frame, 아니면 원본 frame)
  initCostmap(costmap, hdr);

  // 원점 주변 free (옵션)
  {
    int ox, oy;
    worldToGrid(costmap, 0.0f, 0.0f, ox, oy);
    int r = 3;
    for (int y = oy - r; y <= oy + r; ++y) {
      for (int x = ox - r; x <= ox + r; ++x) {
        int idx = gridIndex(costmap, x, y);
        if (idx >= 0) costmap.data[idx] = free_cost;
      }
    }
  }

  // 9) for each detection -> if cav -> publish markers & paint costmap
  for (const auto &d : dets) {
    if (!isCavCandidate(d)) continue;

    Eigen::Vector3f center = 0.5f * (d.min_pt + d.max_pt);

    geometry_msgs::Pose pose;
    pose.position.x = center.x();
    pose.position.y = center.y();
    pose.position.z = center.z();
    pose.orientation.w = 1.0;
    centers_msg.poses.push_back(pose);

    // AABB box marker (green)
    visualization_msgs::Marker box;
    box.header = hdr;
    box.ns = "cav_aabb";
    box.id = cav_id;
    box.type = visualization_msgs::Marker::CUBE;
    box.action = visualization_msgs::Marker::ADD;
    box.pose = pose;
    box.scale.x = std::max(0.05f, d.size.x());
    box.scale.y = std::max(0.05f, d.size.y());
    box.scale.z = std::max(0.05f, d.size.z());
    box.color.r = 0.0f; box.color.g = 1.0f; box.color.b = 0.0f;
    box.color.a = box_alpha;
    box.lifetime = ros::Duration(marker_lifetime);
    boxes_msg.markers.push_back(box);

    // center sphere marker (yellow)
    visualization_msgs::Marker sph;
    sph.header = hdr;
    sph.ns = "cav_center_sphere";
    sph.id = cav_id;
    sph.type = visualization_msgs::Marker::SPHERE;
    sph.action = visualization_msgs::Marker::ADD;
    sph.pose = pose;
    sph.scale.x = center_sphere_diam;
    sph.scale.y = center_sphere_diam;
    sph.scale.z = center_sphere_diam;
    sph.color.r = 1.0f; sph.color.g = 1.0f; sph.color.b = 0.0f;
    sph.color.a = 0.8f;
    sph.lifetime = ros::Duration(marker_lifetime);
    center_spheres_msg.markers.push_back(sph);

    // cav points (red) for debug
    if (d.id >= 0 && d.id < (int)clusters.size()) {
      for (int idx : clusters[d.id].indices) {
        if (idx < 0 || idx >= (int)ransac_cloud->size()) continue;
        const auto &src = ransac_cloud->points[idx];
        pcl::PointXYZRGB p;
        p.x = src.x; p.y = src.y; p.z = src.z;
        p.r = 255; p.g = 0; p.b = 0;
        cav_colored->points.push_back(p);
      }
    }

    // ---- paint costmap with AABB (XY only) ----
    // marker는 중심/스케일 기반이므로, costmap에는 min/max로 채우면 됨
    float min_x = d.min_pt.x();
    float min_y = d.min_pt.y();
    float max_x = d.max_pt.x();
    float max_y = d.max_pt.y();

    // “초록 박스 영역을 costmap에 그대로” → green cost를 사용
    paintAABB(costmap, min_x, min_y, max_x, max_y, cav_cost_green, inflation_m);

    cav_id++;
  }

  // publish cav cloud
  cav_colored->width = (uint32_t)cav_colored->points.size();
  cav_colored->height = 1;
  {
    sensor_msgs::PointCloud2 cav_msg;
    pcl::toROSMsg(*cav_colored, cav_msg);
    cav_msg.header = hdr;
    pub_cluster_cav.publish(cav_msg);
  }

  // publish markers & centers
  pub_cav_aabb_centers.publish(centers_msg);
  pub_cav_aabb_boxes.publish(boxes_msg);
  pub_cav_aabb_center_spheres.publish(center_spheres_msg);

  // publish costmap
  pub_costmap.publish(costmap);
}

// ------------------------------
// main
// ------------------------------
int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar_aabb_costmap_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // load params
  pnh.param("lidar_topic", lidar_topic, lidar_topic);
  pnh.param("costmap_topic", costmap_topic, costmap_topic);

  pnh.param("use_tf", use_tf, use_tf);
  pnh.param("require_tf", require_tf, require_tf);
  pnh.param("costmap_frame", costmap_frame, costmap_frame);

  pnh.param("map_resolution", map_resolution, map_resolution);
  pnh.param("map_width", map_width, map_width);
  pnh.param("map_height", map_height, map_height);
  int tmp;
  tmp = (int)unknown_cost; pnh.param("unknown_cost", tmp, (int)unknown_cost); unknown_cost = (int8_t)tmp;
  tmp = (int)free_cost;    pnh.param("free_cost", tmp, (int)free_cost);       free_cost = (int8_t)tmp;
  tmp = (int)obstacle_cost;pnh.param("obstacle_cost", tmp, (int)obstacle_cost);obstacle_cost=(int8_t)tmp;

  pnh.param("inflation_m", inflation_m, inflation_m);

  pnh.param("min_height", min_height, min_height);
  pnh.param("max_height", max_height, max_height);
  pnh.param("lidar_range", lidar_range, lidar_range);
  pnh.param("voxel_leaf", voxel_leaf, voxel_leaf);

  pnh.param("ego_exclusion", ego_exclusion, ego_exclusion);

  pnh.param("ransac_dist_thresh", ransac_dist_thresh, ransac_dist_thresh);
  pnh.param("ransac_eps_angle_deg", ransac_eps_angle_deg, ransac_eps_angle_deg);
  pnh.param("ransac_max_iter", ransac_max_iter, ransac_max_iter);

  pnh.param("cluster_tolerance", cluster_tolerance, cluster_tolerance);
  pnh.param("cluster_min_size", cluster_min_size, cluster_min_size);
  pnh.param("cluster_max_size", cluster_max_size, cluster_max_size);

  pnh.param("cav_min_points", cav_min_points, cav_min_points);
  pnh.param("cav_dx_min", cav_dx_min, cav_dx_min);
  pnh.param("cav_dx_max", cav_dx_max, cav_dx_max);
  pnh.param("cav_dy_min", cav_dy_min, cav_dy_min);
  pnh.param("cav_dy_max", cav_dy_max, cav_dy_max);
  pnh.param("cav_dz_min", cav_dz_min, cav_dz_min);
  pnh.param("cav_dz_max", cav_dz_max, cav_dz_max);

  tmp = (int)cav_cost_green; pnh.param("cav_cost_green", tmp, (int)cav_cost_green); cav_cost_green = (int8_t)tmp;
  tmp = (int)cav_cost_red;   pnh.param("cav_cost_red", tmp, (int)cav_cost_red);     cav_cost_red   = (int8_t)tmp;

  pnh.param("box_alpha", box_alpha, box_alpha);
  pnh.param("marker_lifetime", marker_lifetime, marker_lifetime);
  pnh.param("center_sphere_diam", center_sphere_diam, center_sphere_diam);

  // TF init
  tfBufferPtr.reset(new tf2_ros::Buffer());
  tfListenerPtr.reset(new tf2_ros::TransformListener(*tfBufferPtr));

  // publishers
  pub_costmap = nh.advertise<nav_msgs::OccupancyGrid>(costmap_topic, 1);
  pub_ransac = nh.advertise<sensor_msgs::PointCloud2>("/ransac", 1);
  pub_cluster_all = nh.advertise<sensor_msgs::PointCloud2>("/cluster", 1);
  pub_cluster_cav = nh.advertise<sensor_msgs::PointCloud2>("/cluster_cav", 1);

  pub_cav_aabb_centers = nh.advertise<geometry_msgs::PoseArray>("/cav_aabb_centers", 1);
  pub_cav_aabb_boxes = nh.advertise<visualization_msgs::MarkerArray>("/cav_aabb_boxes", 1);
  pub_cav_aabb_center_spheres = nh.advertise<visualization_msgs::MarkerArray>("/cav_aabb_center_spheres", 1);

  // subscriber
  ros::Subscriber sub = nh.subscribe(lidar_topic, 1, lidarCallback);

  ROS_INFO("lidar_aabb_costmap_node started");
  ROS_INFO(" - lidar_topic: %s", lidar_topic.c_str());
  ROS_INFO(" - costmap_topic: %s", costmap_topic.c_str());
  ROS_INFO(" - use_tf: %s, costmap_frame: %s", use_tf ? "true":"false", costmap_frame.c_str());
  ROS_INFO(" - map: %.1fm x %.1fm, res=%.2fm", map_width, map_height, map_resolution);
  ROS_INFO(" - height: [%.2f, %.2f], range=%.1f, voxel=%.2f", min_height, max_height, lidar_range, voxel_leaf);
  ROS_INFO(" - cluster tol=%.2f, min=%d, max=%d", cluster_tolerance, cluster_min_size, cluster_max_size);
  ROS_INFO(" - cav dims dx[%.2f,%.2f], dy[%.2f,%.2f], dz[%.2f,%.2f], min_pts=%d",
           cav_dx_min,cav_dx_max,cav_dy_min,cav_dy_max,cav_dz_min,cav_dz_max,cav_min_points);

  ros::spin();
  return 0;
}

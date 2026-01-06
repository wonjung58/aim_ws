#pragma once
#include <string>
#include <cstdint>
#include <ros/ros.h>

struct LidarAabbParams {
// ------------------------------
// Params
// ------------------------------
std::string lidar_topic = "/lidar3D";
std::string costmap_topic = "/costmap_from_lidar"; //OccupancyGrid publish topic

bool use_tf = true;                 // TF로 cloud를 costmap_frame으로 변환할지
bool require_tf = true;             // TF 실패하면 프레임 스킵할지
std::string costmap_frame = "base_link"; // costmap/marker publish frame // costmap 좌표가 차량 기준(로컬 좌표)

// costmap params (meters)
float map_resolution = 0.05f;  // grid 한 칸 크기 (m)  0.05m = 5cm 작을수록 정밀하지만, 셀 개수 증가 → 계산량 증가
float map_width = 50.0f;       // costmap의 실제 물리 크기 (m)
float map_height = 50.0f;      // 50×50m 로컬맵이면, base_link 기준으로 보통 -25~+25m 범위를 커버
int8_t unknown_cost = -1;      // OccupancyGrid: unknown=-1 권장
int8_t free_cost = 0;          // free
int8_t obstacle_cost = 100;    // occupied

float inflation_m = 0.0f;      // AABB 박스를 costmap에 칠할 때 주변을 추가로 두껍게(inflation) 칠하는 값 -> AABB 박스를 costmap에 칠할 때 주변을 추가로 두껍게(inflation) 칠하는 값

// lidar prefilter
// float min_height = -3.0f;
// float max_height = 5.0f;
// float lidar_range = 25.0f;

// ego 제거 ROI (passthrough) -> 차체 x, y 값에 따라 세밀한 파라미터 조정 필요
float ego_xmin = 2.0f; // |x|<=2 and |y|<=2 영역 제거 ==> 자차 차체/센서 브라켓/지붕 포인트 때문에 생기는 “가짜 클러스터” 방지
float ego_xmax = 2.0f; 
float ego_ymin = 2.0f;
float ego_ymax = 2.0f; 

// voxel downsample
float voxel_leaf = 0.10f;

// RANSAC
float ransac_dist_thresh = 0.30f; // 평면에서 0.30m 이내면 지면(inlier)로 간주
float ransac_eps_angle_deg = 10.0f; // 지면 평면의 법선이 z축에서 10도 이내면 지면으로 인정 ==> 약간 기울어진 도로도 지면으로 인식 / 경사로/언덕이 있으면 조금 더 키워야 할 수도 있음(예: 15~20)
int ransac_max_iter = 200; // RANSAC 반복 횟수 (크면 안정적이지만 느려짐)

// clustering
float cluster_tolerance = 0.4f; // 점과 점이 0.4m 이내면 같은 클러스터로 연결
int cluster_min_size = 20;
int cluster_max_size = 2000; // 너무 작은 덩어리(노이즈) 제거 / 너무 큰 덩어리(벽/지형) 제거

// vehicle 후보 조건
int cav_min_points = 10; // 차량 후보로 인정할 최소 점 개수
float cav_dx_min = 0.3f, cav_dx_max = 8.0f; // AABB의 x방향 길이 범위 (m)
float cav_dy_min = 0.2f, cav_dy_max = 4.0f; // AABB의 y방향 길이 범위 (m)
float cav_dz_min = 0.1f, cav_dz_max = 3.0f; // AABB의 z방향 길이 범위 (m) => dz로 “평평한 바닥 노이즈/큰 구조물” 거르는데 도움 됨

// cost 값 (박스 컬러별)
int8_t cav_cost_green = 75; // 차량 후보 박스를 costmap에 칠할 때 쓸 값(중간 위험)
int8_t cav_cost_red   = 100; // 차량 후보 박스를 costmap에 칠할 때 쓸 값(위험)

// marker(rviz params)
float box_alpha = 0.25f; // AABB 박스 투명도
float marker_lifetime = 0.2f; // marker 수명 (초)
float center_sphere_diam = 0.3f; // AABB 센터 구체 지름 (m)
};

static void loadParams(ros::NodeHandle& pnh, LidarAabbParams& P)
{
  // string
  pnh.param("lidar_topic",    P.lidar_topic,    P.lidar_topic);
  pnh.param("costmap_topic",  P.costmap_topic,  P.costmap_topic);
  pnh.param("costmap_frame",  P.costmap_frame,  P.costmap_frame);

  // bool
  pnh.param("use_tf",     P.use_tf,     P.use_tf);
  pnh.param("require_tf", P.require_tf, P.require_tf);

  // float/double
  pnh.param("map_resolution", P.map_resolution, P.map_resolution);
  pnh.param("map_width",      P.map_width,      P.map_width);
  pnh.param("map_height",     P.map_height,     P.map_height);

  pnh.param("inflation_m",    P.inflation_m,    P.inflation_m);

  // pnh.param("min_height",     P.min_height,     P.min_height);
  // pnh.param("max_height",     P.max_height,     P.max_height);
  // pnh.param("lidar_range",    P.lidar_range,    P.lidar_range);
  // pnh.param("ego_exclusion",  P.ego_exclusion,  P.ego_exclusion);
  pnh.param("ego_xmin",       P.ego_xmin,       P.ego_xmin);
  pnh.param("ego_xmax",       P.ego_xmax,       P.ego_xmax);
  pnh.param("ego_ymin",       P.ego_ymin,       P.ego_ymin);
  pnh.param("ego_ymax",       P.ego_ymax,       P.ego_ymax);  

  pnh.param("voxel_leaf", P.voxel_leaf, P.voxel_leaf);
  
  pnh.param("ransac_dist_thresh",   P.ransac_dist_thresh,   P.ransac_dist_thresh);
  pnh.param("ransac_eps_angle_deg", P.ransac_eps_angle_deg, P.ransac_eps_angle_deg);
  pnh.param("ransac_max_iter",      P.ransac_max_iter,      P.ransac_max_iter);

  pnh.param("cluster_tolerance", P.cluster_tolerance, P.cluster_tolerance);
  pnh.param("cluster_min_size",  P.cluster_min_size,  P.cluster_min_size);
  pnh.param("cluster_max_size",  P.cluster_max_size,  P.cluster_max_size);

  pnh.param("cav_min_points", P.cav_min_points, P.cav_min_points);
  pnh.param("cav_dx_min",     P.cav_dx_min,     P.cav_dx_min);
  pnh.param("cav_dx_max",     P.cav_dx_max,     P.cav_dx_max);
  pnh.param("cav_dy_min",     P.cav_dy_min,     P.cav_dy_min);
  pnh.param("cav_dy_max",     P.cav_dy_max,     P.cav_dy_max);
  pnh.param("cav_dz_min",     P.cav_dz_min,     P.cav_dz_min);
  pnh.param("cav_dz_max",     P.cav_dz_max,     P.cav_dz_max);

  // int8_t는 rosparam에서 바로 못 읽는 경우가 많아서 int로 받고 캐스팅
  int tmp = 0;

  tmp = (int)P.unknown_cost; pnh.param("unknown_cost", tmp, tmp); P.unknown_cost = (int8_t)tmp;
  tmp = (int)P.free_cost;    pnh.param("free_cost",    tmp, tmp); P.free_cost    = (int8_t)tmp;
  tmp = (int)P.obstacle_cost;pnh.param("obstacle_cost",tmp, tmp); P.obstacle_cost= (int8_t)tmp;

  tmp = (int)P.cav_cost_green; pnh.param("cav_cost_green", tmp, tmp); P.cav_cost_green = (int8_t)tmp;
  tmp = (int)P.cav_cost_red;   pnh.param("cav_cost_red",   tmp, tmp); P.cav_cost_red   = (int8_t)tmp;

  // marker
  pnh.param("box_alpha",          P.box_alpha,          P.box_alpha);
  pnh.param("marker_lifetime",    P.marker_lifetime,    P.marker_lifetime);
  pnh.param("center_sphere_diam", P.center_sphere_diam, P.center_sphere_diam);
}
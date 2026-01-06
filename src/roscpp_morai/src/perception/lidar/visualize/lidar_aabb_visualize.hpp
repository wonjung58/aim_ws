#ifndef LIDAR_AABB_VISUALIZE_HPP
#define LIDAR_AABB_VISUALIZE_HPP

#include <nav_msgs/OccupancyGrid.h>
#include <algorithm>
#include <cmath>

// Helper: clamp int
static inline int clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

// Paint AABB on costmap
static void paintAABB(nav_msgs::OccupancyGrid &cm,
                      float min_x, float min_y, float max_x, float max_y,
                      int8_t value, float inflation = 0.0f)
{
  if (min_x > max_x) std::swap(min_x, max_x);
  if (min_y > max_y) std::swap(min_y, max_y);

  // inflation 적용
  min_x -= inflation; min_y -= inflation;
  max_x += inflation; max_y += inflation;

  int gx0, gy0, gx1, gy1;
  if (!worldToGrid(cm, min_x, min_y, gx0, gy0)) {
    gx0 = (int)std::floor((min_x - cm.info.origin.position.x) / cm.info.resolution);
    gy0 = (int)std::floor((min_y - cm.info.origin.position.y) / cm.info.resolution);
  }
  if (!worldToGrid(cm, max_x, max_y, gx1, gy1)) {
    gx1 = (int)std::floor((max_x - cm.info.origin.position.x) / cm.info.resolution);
    gy1 = (int)std::floor((max_y - cm.info.origin.position.y) / cm.info.resolution);
  }

  if (gx0 > gx1) std::swap(gx0, gx1);
  if (gy0 > gy1) std::swap(gy0, gy1);

  gx0 = clampi(gx0, 0, (int)cm.info.width  - 1);
  gx1 = clampi(gx1, 0, (int)cm.info.width  - 1);
  gy0 = clampi(gy0, 0, (int)cm.info.height - 1);
  gy1 = clampi(gy1, 0, (int)cm.info.height - 1);

  const int w = (int)cm.info.width;
  for (int y = gy0; y <= gy1; ++y) {
    for (int x = gx0; x <= gx1; ++x) {
      int idx = y * w + x;
      if (idx >= 0 && idx < (int)cm.data.size()) {
        cm.data[idx] = std::max(cm.data[idx], value);
      }
    }
  }
}

// print AABB
void printLidarAABB_statistics(const LidarAabbParams& g_params) {
    ROS_INFO("lidar_aabb_costmap_node started");
    ROS_INFO(" - lidar_topic: %s", g_params.lidar_topic.c_str());
    ROS_INFO(" - costmap_topic: %s", g_params.costmap_topic.c_str());
    ROS_INFO(" - use_tf: %s, require_tf: %s, costmap_frame: %s",
            g_params.use_tf ? "true":"false",
            g_params.require_tf ? "true":"false",
            g_params.costmap_frame.c_str());
    ROS_INFO(" - map: %.1fm x %.1fm, res=%.2fm", g_params.map_width, g_params.map_height, g_params.map_resolution);
    // ROS_INFO(" - height: [%.2f, %.2f], range=%.1f, voxel=%.2f", g_params.min_height, g_params.max_height, g_params.lidar_range, g_params.voxel_leaf);
    ROS_INFO(" - cluster tol=%.2f, min=%d, max=%d", g_params.cluster_tolerance, g_params.cluster_min_size, g_params.cluster_max_size);
    ROS_INFO(" - cav dims dx[%.2f,%.2f], dy[%.2f,%.2f], dz[%.2f,%.2f], min_pts=%d",
            g_params.cav_dx_min,g_params.cav_dx_max,g_params.cav_dy_min,g_params.cav_dy_max,g_params.cav_dz_min,g_params.cav_dz_max,g_params.cav_min_points);
            // cav AABB 조건  
}

#endif // LIDAR_AABB_VISUALIZE_HPP
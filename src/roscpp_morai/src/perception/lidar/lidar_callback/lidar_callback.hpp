void lidarCallback(const sensor_msgs::PointCloud2ConstPtr &in_msg)
{
  if (!ros::ok()) return;

  const std::string target_frame = normalizeFrame(g_params.costmap_frame);
  const std::string src_frame_raw = in_msg->header.frame_id;
  const std::string source_frame = normalizeFrame(src_frame_raw);

  // 1) TF transform to costmap_frame
  sensor_msgs::PointCloud2 cloud_tf; // 맨 처음에 생성 -> 형변환
  std_msgs::Header hdr = in_msg->header;
  hdr.frame_id = source_frame; // normalize

  bool in_target_frame = (hdr.frame_id == target_frame);

  if (g_params.use_tf && !in_target_frame) {
    try {
      // 시뮬에서 stamp 기반 extrapolation이 자주 나서 latest(0) 사용
      geometry_msgs::TransformStamped tf =
        tfBufferPtr->lookupTransform(target_frame, hdr.frame_id, ros::Time(0), ros::Duration(0.05));

      tf2::doTransform(*in_msg, cloud_tf, tf); 
      // ---------- 함수 선언 안 보임 ----------

      // header 강제 통일
      hdr = cloud_tf.header;
      hdr.frame_id = target_frame;
      hdr.stamp = in_msg->header.stamp;

      cloud_tf.header = hdr;
      in_target_frame = true;
    } catch (const std::exception &e) {
      ROS_WARN_THROTTLE(1.0, "TF failed (%s -> %s): %s",
                        hdr.frame_id.c_str(), target_frame.c_str(), e.what());
      if (g_params.require_tf) return;

      // fallback(비추천): frame 그대로 처리
      cloud_tf = *in_msg;
      hdr = cloud_tf.header;
      hdr.frame_id = normalizeFrame(hdr.frame_id);
      cloud_tf.header = hdr;
      in_target_frame = (hdr.frame_id == target_frame);
    }
  } else {
    cloud_tf = *in_msg;
    hdr = cloud_tf.header;
    hdr.frame_id = normalizeFrame(hdr.frame_id);
    hdr.stamp = in_msg->header.stamp;
    cloud_tf.header = hdr;
    in_target_frame = (hdr.frame_id == target_frame);
  }

  // 아래 publish들은 “base_link 기준”을 목표로 함.
  // require_tf=true면 항상 in_target_frame=true라서 안전.
  if (g_params.use_tf && g_params.require_tf && !in_target_frame) return;

  // 2) ROS -> PCL
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(cloud_tf, *cloud);
  if (cloud->empty()) return;

  // 3) filters
  // filterByHeight(cloud, g_params.min_height, g_params.max_height);
  // filterByRange(cloud, g_params.lidar_range);
  // if (cloud->empty()) return;

  // 3) remove ego ROI
  pcl::PointCloud<pcl::PointXYZI>::Ptr roi_cloud = 
  removeEgoROI(cloud, g_params.ego_xmin, g_params.ego_xmax, g_params.ego_ymin, g_params.ego_ymax);
  if (roi_cloud->empty()) return;

  // 4) voxel downsample
  voxelDownsample(roi_cloud, g_params.voxel_leaf);
  if (roi_cloud->empty()) return;

  // 5) RANSAC ground removal
  pcl::PointCloud<pcl::PointXYZI>::Ptr ransac_cloud =
    ransacRemoveGround(roi_cloud, g_params.ransac_dist_thresh, g_params.ransac_eps_angle_deg, g_params.ransac_max_iter);

  // debug publish: ransac cloud (frame/stamp 통일)
  {
    sensor_msgs::PointCloud2 ransac_out;
    pcl::toROSMsg(*ransac_cloud, ransac_out);
    ransac_out.header = hdr;
    ransac_out.header.frame_id = target_frame;
    ransac_out.header.stamp = in_msg->header.stamp;
    pub_ransac.publish(ransac_out);
  }

  if (ransac_cloud->empty()) return;

  // 6) clustering
  std::vector<pcl::PointIndices> clusters =
    euclideanCluster(ransac_cloud, g_params.cluster_tolerance, g_params.cluster_min_size, g_params.cluster_max_size);

  // publish all colored clusters_indices
  {
    std_msgs::Header chdr = hdr;
    chdr.frame_id = target_frame;
    chdr.stamp = in_msg->header.stamp;
    publishColoredClustersAll(ransac_cloud, clusters, chdr);
  }

  // 7) detections
  std::vector<Detection> cluster_info = clusterInfo(ransac_cloud, clusters, hdr.stamp);
  // clusters -> clusters_indices 로 바꾸고싶

  // messages to publish (frame/stamp 통일)
  geometry_msgs::PoseArray centers_msg;
  centers_msg.header.frame_id = target_frame;
  centers_msg.header.stamp = in_msg->header.stamp;

  visualization_msgs::MarkerArray boxes_msg;
  visualization_msgs::MarkerArray center_spheres_msg;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cav_colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  cav_colored->is_dense = true;

  int cav_id = 0;


  // ========================================================================================
  // costmap & markers for CAV candidates
  // ========================================================================================

  // 8) init costmap (local around origin)
  nav_msgs::OccupancyGrid costmap;
  initCostmap(costmap, centers_msg.header);
  costmap.header.frame_id = target_frame;              // 강제
  costmap.header.stamp = in_msg->header.stamp;         // 강제

  // 원점 주변 free (옵션)
  {
    int ox, oy;
    worldToGrid(costmap, 0.0f, 0.0f, ox, oy);
    int r = 3;
    for (int y = oy - r; y <= oy + r; ++y) {
      for (int x = ox - r; x <= ox + r; ++x) {
        int idx = gridIndex(costmap, x, y);
        if (idx >= 0) costmap.data[idx] = g_params.free_cost;
      }
    }
  }

  // 9) for each detection -> if cav -> publish markers & paint costmap
  for (const auto &d : cluster_info) {
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
    box.header.frame_id = target_frame;
    box.header.stamp = in_msg->header.stamp;
    box.ns = "cav_aabb";
    box.id = cav_id;
    box.type = visualization_msgs::Marker::CUBE;
    box.action = visualization_msgs::Marker::ADD;
    box.pose = pose;
    box.scale.x = std::max(0.05f, d.size.x());
    box.scale.y = std::max(0.05f, d.size.y());
    box.scale.z = std::max(0.05f, d.size.z());
    box.color.r = 0.0f; box.color.g = 1.0f; box.color.b = 0.0f;
    box.color.a = g_params.box_alpha;
    box.lifetime = ros::Duration(g_params.marker_lifetime);
    boxes_msg.markers.push_back(box);

    // center sphere marker (yellow)
    visualization_msgs::Marker sph;
    sph.header.frame_id = target_frame;
    sph.header.stamp = in_msg->header.stamp;
    sph.ns = "cav_center_sphere";
    sph.id = cav_id;
    sph.type = visualization_msgs::Marker::SPHERE;
    sph.action = visualization_msgs::Marker::ADD;
    sph.pose = pose;
    sph.scale.x = g_params.center_sphere_diam;
    sph.scale.y = g_params.center_sphere_diam;
    sph.scale.z = g_params.center_sphere_diam;
    sph.color.r = 1.0f; sph.color.g = 1.0f; sph.color.b = 0.0f;
    sph.color.a = 0.8f;
    sph.lifetime = ros::Duration(g_params.marker_lifetime);
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
    float min_x = d.min_pt.x();
    float min_y = d.min_pt.y();
    float max_x = d.max_pt.x();
    float max_y = d.max_pt.y();
    paintAABB(costmap, min_x, min_y, max_x, max_y, g_params.cav_cost_green, g_params.inflation_m);

    cav_id++;
  }

  // publish cav cloud (frame/stamp 통일)
  cav_colored->width = (uint32_t)cav_colored->points.size();
  cav_colored->height = 1;
  {
    sensor_msgs::PointCloud2 cav_msg;
    pcl::toROSMsg(*cav_colored, cav_msg);
    cav_msg.header.frame_id = target_frame;
    cav_msg.header.stamp = in_msg->header.stamp;
    pub_cluster_cav.publish(cav_msg);
  }

  // publish markers & centers
  pub_cav_aabb_centers.publish(centers_msg);
  pub_cav_aabb_boxes.publish(boxes_msg);
  pub_cav_aabb_center_spheres.publish(center_spheres_msg);

  // publish costmap
  pub_costmap.publish(costmap);
}

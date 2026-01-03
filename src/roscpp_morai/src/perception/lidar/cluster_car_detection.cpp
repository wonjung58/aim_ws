#include <iostream>
#include <cmath>
#include <limits>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
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

using namespace std;
using namespace ros;

Publisher pub_ransac;
Publisher pub_cluster;
Publisher pub_cluster_all;
Publisher pub_cluster_cav;

// (추가) cav AABB 중심 퍼블리셔
Publisher pub_cav_aabb_centers;

// (추가) cav AABB 박스(마커) 퍼블리셔
Publisher pub_cav_aabb_boxes;

// (추가) cav AABB 중심점 구 마커 퍼블리셔
Publisher pub_cav_aabb_center_spheres;


// ------------------------------
// detection struct
// ------------------------------
struct detection {
    int det_id;
    Time stamp;

    Eigen::Vector3f centroid;

    Eigen::Vector3f min_pt;
    Eigen::Vector3f max_pt;
    Eigen::Vector3f size;

    int num_points;
    float range;

    bool is_vehicle_candidate;
};

// ------------------------------
// 클러스터 -> detection 계산
// ------------------------------
vector<detection> calculate_det(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& ransac_cloud,
    const std::vector<pcl::PointIndices>& cluster_indices,
    const ros::Time& stamp)
{
    vector<detection> det;
    det.reserve(cluster_indices.size());

    for (int i = 0; i < (int)cluster_indices.size(); ++i) {

        const vector<int>& vector_idx = cluster_indices[i].indices;
        if (vector_idx.empty()) continue;

        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

        float min_x = std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        int count = 0;

        for (int j = 0; j < (int)vector_idx.size(); ++j) {
            int idx = vector_idx[j];
            if (idx < 0 || idx >= (int)ransac_cloud->size()) continue;

            const pcl::PointXYZI& p = ransac_cloud->points[idx];

            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z;

            if (p.x < min_x) min_x = p.x;
            if (p.y < min_y) min_y = p.y;
            if (p.z < min_z) min_z = p.z;

            if (p.x > max_x) max_x = p.x;
            if (p.y > max_y) max_y = p.y;
            if (p.z > max_z) max_z = p.z;

            ++count;
        }

        if (count == 0) continue;

        const float cx = static_cast<float>(sum_x / count);
        const float cy = static_cast<float>(sum_y / count);
        const float cz = static_cast<float>(sum_z / count);

        detection d;
        d.det_id = i;
        d.stamp = stamp;
        d.num_points = count;

        d.centroid = Eigen::Vector3f(cx, cy, cz);

        d.min_pt = Eigen::Vector3f(min_x, min_y, min_z);
        d.max_pt = Eigen::Vector3f(max_x, max_y, max_z);
        d.size   = d.max_pt - d.min_pt;

        d.range = std::sqrt(cx * cx + cy * cy);

        d.is_vehicle_candidate = true;

        det.push_back(d);
    }

    return det;
}

// ------------------------------
// 차량 후보 판정(튜닝 필요)
// ------------------------------
bool cav_cluster(const detection& d)
{
    const float dx = d.size.x();
    const float dy = d.size.y();
    const float dz = d.size.z();

    ROS_INFO("Detection check - points:%d, size(%.2f,%.2f,%.2f)", 
             d.num_points, dx, dy, dz);

    if (d.num_points < 10) {
        ROS_DEBUG("Rejected: too few points (%d < 10)", d.num_points);
        return false;
    }

    if (dz < 0.1f || dz > 3.0f) {
        ROS_DEBUG("Rejected: bad height (%.2f not in [0.1, 3.0])", dz);
        return false;
    }
    if (dx < 0.3f || dx > 8.0f) {
        ROS_DEBUG("Rejected: bad width (%.2f not in [0.3, 8.0])", dx);
        return false;
    }
    if (dy < 0.2f || dy > 4.0f) {
        ROS_DEBUG("Rejected: bad length (%.2f not in [0.2, 4.0])", dy);
        return false;
    }

    ROS_INFO("✓ Vehicle detected!");
    return true;
}

// ------------------------------
// callback
// ------------------------------
void callback(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
{
    // ROS -> PCL
    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl::fromROSMsg(*input_cloud, pcl_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    *cloud = pcl_cloud;

    // 1) voxel downsample
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    voxel.setInputCloud(cloud);
    float leafsize = 0.1f;
    voxel.setLeafSize(leafsize, leafsize, leafsize);
    voxel.filter(*voxel_cloud);

    // 2) passthrough로 자차 주변 제거
    pcl::PointCloud<pcl::PointXYZI>::Ptr x_inside(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr x_outside(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    float car_size = 2.0f;

    pcl::PassThrough<pcl::PointXYZI> x_filter;
    x_filter.setInputCloud(voxel_cloud);
    x_filter.setFilterFieldName("x");
    x_filter.setFilterLimits(-car_size, car_size);

    x_filter.setFilterLimitsNegative(false);
    x_filter.filter(*x_inside);

    x_filter.setFilterLimitsNegative(true);
    x_filter.filter(*x_outside);

    pcl::PassThrough<pcl::PointXYZI> y_filter;
    y_filter.setInputCloud(x_inside);
    y_filter.setFilterFieldName("y");
    y_filter.setFilterLimits(-car_size, car_size);

    // 중앙(|y|<=car_size) 삭제 -> 바깥만 남김
    y_filter.setFilterLimitsNegative(true);
    y_filter.filter(*roi_cloud);

    *roi_cloud += *x_outside;

    // 3) RANSAC 지면 제거
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.3);
    seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    seg.setEpsAngle(10.0 * M_PI / 180.0);
    seg.setMaxIterations(200);
    seg.setInputCloud(roi_cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        cout << "no model" << endl;
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr ransac_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(roi_cloud);
    extract.setIndices(inliers);

    // outlier(비지면)만 남김
    extract.setNegative(true);
    extract.filter(*ransac_cloud);

    // /ransac publish
    sensor_msgs::PointCloud2 output_cloud;
    pcl::toROSMsg(*ransac_cloud, output_cloud);
    output_cloud.header.frame_id = "lidar_link";
    output_cloud.header.stamp = input_cloud->header.stamp;
    pub_ransac.publish(output_cloud);

    // 4) Euclidean clustering
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
    kdtree->setInputCloud(ransac_cloud);

    vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> euclidean;
    euclidean.setInputCloud(ransac_cloud);
    euclidean.setSearchMethod(kdtree);
    euclidean.setClusterTolerance(0.4);      // ↑ 0.3 → 0.4 (더 관대)
    euclidean.setMinClusterSize(20);         // ↓ 100 → 20 (더 작은 클러스터도 허용)
    euclidean.setMaxClusterSize(2000);       // ↑ 1000 → 2000 (큰 클러스터도 허용)
    euclidean.extract(cluster_indices);

    ROS_INFO_STREAM("---- clusters: " << cluster_indices.size() << " ----");

    // (전체 클러스터 시각화) /cluster
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored->is_dense = true;
    colored->points.reserve(ransac_cloud->points.size());

    int cluster_id = 0;
    for (const auto& indices : cluster_indices)
    {
        uint8_t r = 0, g = 0, b = 0;
        switch (cluster_id % 6) {
            case 0: r = 255; g =  80; b =  80; break;
            case 1: r =  80; g = 255; b =  80; break;
            case 2: r =  80; g =  80; b = 255; break;
            case 3: r = 255; g = 255; b =  80; break;
            case 4: r = 255; g =  80; b = 255; break;
            case 5: r =  80; g = 255; b = 255; break;
        }

        for (int idx : indices.indices)
        {
            const auto& src = ransac_cloud->points[idx];
            pcl::PointXYZRGB p;
            p.x = src.x; p.y = src.y; p.z = src.z;
            p.r = r; p.g = g; p.b = b;
            colored->points.push_back(p);
        }

        cluster_id++;
    }

    colored->width = (uint32_t)colored->points.size();
    colored->height = 1;

    sensor_msgs::PointCloud2 cluster_msg;
    pcl::toROSMsg(*colored, cluster_msg);
    cluster_msg.header.frame_id = "lidar_link";
    cluster_msg.header.stamp = input_cloud->header.stamp;
    pub_cluster.publish(cluster_msg);

    // 5) detection 계산 (AABB/centroid 포함)
    vector<detection> det_all = calculate_det(ransac_cloud, cluster_indices, input_cloud->header.stamp);

    // (추가) cav AABB 중심들 PoseArray
    geometry_msgs::PoseArray centers_msg;
    centers_msg.header = input_cloud->header;
    if (centers_msg.header.frame_id.empty())
        centers_msg.header.frame_id = "lidar_link";

    // (추가) cav AABB 박스 MarkerArray
    visualization_msgs::MarkerArray boxes_msg;

    // (cav만 시각화) /cluster_cav
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cav(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cav->is_dense = true;

    // cav 후보마다 marker id 증가용
    int cav_marker_id = 0;

    for (int i = 0; i < (int)cluster_indices.size(); ++i) {
        bool is_cav = false;
        detection d_i;

        // i번째 클러스터의 detection 찾기
        for (const auto& d : det_all) {
            if (d.det_id == i) {
                d_i = d;
                is_cav = cav_cluster(d);
                break;
            }
        }
        if (!is_cav) continue;

        // --------- AABB center 계산 후 PoseArray에 push ---------
        Eigen::Vector3f aabb_center = 0.5f * (d_i.min_pt + d_i.max_pt);

        geometry_msgs::Pose pose;
        pose.position.x = aabb_center.x();
        pose.position.y = aabb_center.y();
        pose.position.z = aabb_center.z();
        pose.orientation.x = 0.0;
        pose.orientation.y = 0.0;
        pose.orientation.z = 0.0;
        pose.orientation.w = 1.0;
        centers_msg.poses.push_back(pose);
        // -------------------------------------------------------

        // --------- (추가) RViz AABB 박스 Marker(CUBE) 생성 ---------
        visualization_msgs::Marker box;
        box.header = centers_msg.header;            // frame_id/stamp 동일
        box.ns = "cav_aabb";
        box.id = cav_marker_id++;                   // 프레임 내 유니크 id
        box.type = visualization_msgs::Marker::CUBE;
        box.action = visualization_msgs::Marker::ADD;

        box.pose = pose;                            // 중심 좌표

        // 박스 크기(스케일) = AABB size
        // 0이면 RViz에서 안 보이니 최소값 보정(아주 얇게라도)
        box.scale.x = std::max(0.05f, d_i.size.x());
        box.scale.y = std::max(0.05f, d_i.size.y());
        box.scale.z = std::max(0.05f, d_i.size.z());

        // 색/투명도 (RViz는 alpha=0이면 안 보임)
        box.color.r = 0.0f;
        box.color.g = 1.0f;
        box.color.b = 0.0f;
        box.color.a = 0.25f; // 반투명

        // 이전 프레임 잔상 방지: 짧은 lifetime (콜백 주기보다 약간 길게)
        box.lifetime = ros::Duration(0.2);

        boxes_msg.markers.push_back(box);
        // -------------------------------------------------------

        // cav 포인트만 따로 모으기 (현재는 빨강)
        for (int idx : cluster_indices[i].indices)
        {
            const auto& src = ransac_cloud->points[idx];
            pcl::PointXYZRGB p;
            p.x = src.x; p.y = src.y; p.z = src.z;
            p.r = 255; p.g = 0; p.b = 0;
            colored_cav->points.push_back(p);
        }
    }

    // /cluster_cav publish
    colored_cav->width  = (uint32_t)colored_cav->points.size();
    colored_cav->height = 1;

    sensor_msgs::PointCloud2 cav_msg;
    pcl::toROSMsg(*colored_cav, cav_msg);
    cav_msg.header = input_cloud->header;
    if (cav_msg.header.frame_id.empty())
        cav_msg.header.frame_id = "lidar_link";

    pub_cluster_cav.publish(cav_msg);

    // (추가) cav AABB centers publish
    pub_cav_aabb_centers.publish(centers_msg);

    // (추가) cav AABB 중심점 SPHERE 마커 생성 및 퍼블리시
    visualization_msgs::MarkerArray center_spheres_msg;
    int sphere_id = 0;
    for (const auto& pose : centers_msg.poses) {
        visualization_msgs::Marker sphere;
        sphere.header = centers_msg.header;
        sphere.ns = "cav_center_sphere";
        sphere.id = sphere_id++;
        sphere.type = visualization_msgs::Marker::SPHERE;
        sphere.action = visualization_msgs::Marker::ADD;
        
        sphere.pose = pose;
        
        // 구 크기 (지름 0.3m)
        sphere.scale.x = 0.3;
        sphere.scale.y = 0.3;
        sphere.scale.z = 0.3;
        
        // 색: 황색 (밝고 구분 잘됨)
        sphere.color.r = 1.0f;
        sphere.color.g = 1.0f;
        sphere.color.b = 0.0f;
        sphere.color.a = 0.8f;  // 불투명도
        
        sphere.lifetime = ros::Duration(0.2);
        center_spheres_msg.markers.push_back(sphere);
    }
    pub_cav_aabb_center_spheres.publish(center_spheres_msg);

    // (추가) cav AABB boxes publish
    pub_cav_aabb_boxes.publish(boxes_msg);
}

// ------------------------------
// main
// ------------------------------
int main(int argc, char** argv)
{
    init(argc, argv, "func_3D");

    NodeHandle nh;

    Subscriber sub = nh.subscribe("/lidar3D", 1, callback);

    pub_ransac      = nh.advertise<sensor_msgs::PointCloud2>("/ransac", 1);
    pub_cluster     = nh.advertise<sensor_msgs::PointCloud2>("/cluster", 1);
    pub_cluster_all = nh.advertise<sensor_msgs::PointCloud2>("/cluster_all", 1);
    pub_cluster_cav = nh.advertise<sensor_msgs::PointCloud2>("/cluster_cav", 1);

    // (추가) cav 중심 좌표
    pub_cav_aabb_centers = nh.advertise<geometry_msgs::PoseArray>("/cav_aabb_centers", 1);

    // (추가) cav AABB 박스 마커
    pub_cav_aabb_boxes = nh.advertise<visualization_msgs::MarkerArray>("/cav_aabb_boxes", 1);

    // (추가) cav AABB 중심점 구 마커
    pub_cav_aabb_center_spheres = nh.advertise<visualization_msgs::MarkerArray>("/cav_aabb_center_spheres", 1);

    spin();
    return 0;
}

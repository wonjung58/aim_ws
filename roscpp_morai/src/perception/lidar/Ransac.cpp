#include <Global/Global.hpp>

// Ransac 수정해야. 

void Ransac (Lidar& st_Lidar)
{
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);
    seg.setMaxIterations(100);

    seg.setInputCloud(st_Lidar.voxel_cloud);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(st_Lidar.voxel_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true); // 지면 제거
    extract.filter(*st_Lidar.ransac_cloud);
}
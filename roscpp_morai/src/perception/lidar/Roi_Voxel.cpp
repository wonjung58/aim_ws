#include <Global/Global.hpp>

// parameter 도 Global.hpp 에 struct LidarParam 으로 선언해서 관리하도록
// PassThrough 클래스 사용 안 되고 있는 듯 하다. 

void Roi (Lidar& st_Lidar)
{
    pcl::PointCloud <pcl::PointXYZI>::Ptr x_inside (new pcl::PointCloud <pcl::PointXYZI>);
    pcl::PointCloud <pcl::PointXYZI>::Ptr x_outside (new pcl::PointCloud <pcl::PointXYZI>);
    pcl::PointCloud <pcl::PointXYZI>::Ptr vehicle_before_filter_cloud(new pcl::PointCloud <pcl::PointXYZI>);
    pcl::PointCloud <pcl::PointXYZI>::Ptr vehicle_after_filter_cloud(new pcl::PointCloud <pcl::PointXYZI>);
    pcl::PassThrough <pcl::PointXYZI> x_filter; 

    *vehicle_before_filter_cloud = *st_Lidar.raw_cloud;

    float car_size = 2;

    x_filter.setInputCloud (vehicle_before_filter_cloud);
    x_filter.setFilterFieldName("x");
    x_filter.setFilterLimits(-car_size, car_size);
    x_filter.setFilterLimitsNegative(false); 
    x_filter.filter(*x_inside);

    x_filter.setFilterLimitsNegative(true);
    x_filter.filter(*x_outside);

    pcl::PassThrough <pcl::PointXYZI> y_filter;
    y_filter.setInputCloud (x_inside); 
    y_filter.setFilterFieldName("y"); 
    y_filter.setFilterLimits(-car_size, car_size);
    y_filter.setFilterLimitsNegative(true); 
    y_filter.filter(*vehicle_after_filter_cloud); 

    *vehicle_after_filter_cloud += *x_outside;
    *st_Lidar.roi_cloud = *vehicle_after_filter_cloud;
}

void Voxel (Lidar& st_Lidar)
{
    pcl::VoxelGrid<pcl::PointXYZI> voxel;

    voxel.setInputCloud(st_Lidar.roi_cloud);
    float leafsize = 0.1f; // voxel 크기가 0.1f : 10cm 정육면체
    voxel.setLeafSize(leafsize, leafsize, leafsize); // x, y, z 축
    voxel.filter(*st_Lidar.voxel_cloud);
}
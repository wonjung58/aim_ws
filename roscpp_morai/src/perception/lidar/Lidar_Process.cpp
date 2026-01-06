#include <Global/Global.hpp>
#include <lidar/Roi_Voxel.cpp>
#include

// LidarProcess 구성

// 1. global 구조체 Lidar 선언
// 2. MakePointCloudFromRaw 함수 생성 (LidarProcess.cpp)
// 3. LidarProcess(callback) 함수 생성
// 4. main 함수에서 sub, pub, spin()

// input_cloud 를 Global.hpp 의 Lidar 구조체에 넣어야 할까. 
// publisher는 기존과 다르게 전역변수 선언 x

// ------------------------------
// global variables_struct 객체
// ------------------------------

Lidar st_Lidar;

void MakePointCloudFromRaw(Lidar& st_Lidar)
{
    pcl::PointCloud <pcl::PointXYZI> pcl_cloud;

    // pcl::PointCloud <pcl::PointXYZI>::Ptr raw_cloud(new pcl::PointCloud <pcl::PointXYZI>);

    pcl::fromROSMsg(*st_Lidar.input_cloud, pcl_cloud);

    *(st_Lidar.raw_cloud) = pcl_cloud;
}

int main(int argc, char** argv)
{
    //  main 함수에서 sub, pub, spin()
    // LidarProcess(callback) 함수는 위에 void로 생성

    ros::init(argc, argv, "LidarProcess");
    ros::NodeHandle nh;

    Subscriber sub = nh.subscribe("/lidar3D", 1, callback);

    pub_ransac = nh.advertise<sensor_msgs::PointCloud2>("/ransac", 1);
    pub_cluster = nh.advertise<sensor_msgs::PointCloud2>("/cluster", 1);

    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        MakePointCloudFromRaw(st_Lidar);

        Roi(st_Lidar);
        Voxel(st_Lidar);
        Ransac(st_Lidar);
        Euclidean(st_Lidar);
        Lshapefilter(st_Lidar);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
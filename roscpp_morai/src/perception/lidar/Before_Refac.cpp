#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
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
#include <pcl/common/centroid.h>  
#include <pcl/common/common.h>
#include <laser_geometry/laser_geometry.h>

using namespace std;
using namespace ros;

Publisher pub_ransac;
Publisher pub_cluster;

void callback(const sensor_msgs::PointCloud2ConstPtr& input_cloud) {
    pcl::PointCloud <pcl::PointXYZI> pcl_cloud;

    pcl::PointCloud <pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZI>);
    
    pcl::fromROSMsg(*input_cloud, pcl_cloud);

    *cloud = pcl_cloud;

    // ------------------------------------------------------------------------------------------

    // 1. voxel
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    pcl::PointCloud <pcl::PointXYZI>::Ptr voxel_cloud(new pcl::PointCloud <pcl::PointXYZI>);

    voxel.setInputCloud(cloud);
    float leafsize = 0.1f; // voxel 크기가 0.1f : 10cm 정육면체
    voxel.setLeafSize(leafsize, leafsize, leafsize); // x, y, z 축
    voxel.filter(*voxel_cloud);

    // ------------------------------------------------------------------------------------------

    // passthrough 필터로 차체제거

    // 2. PassThrough 필터 
    pcl::PointCloud <pcl::PointXYZI>::Ptr x_inside (new pcl::PointCloud <pcl::PointXYZI>);
    pcl::PointCloud <pcl::PointXYZI>::Ptr x_outside (new pcl::PointCloud <pcl::PointXYZI>);
    // pcl::PointCloud <pcl::PointXYZI>::Ptr y_outside(new pcl::PointCloud <pcl::PointXYZI>);
    pcl::PointCloud <pcl::PointXYZI>::Ptr roi_cloud(new pcl::PointCloud <pcl::PointXYZI>);


    float car_size = 2;

    // 중앙 부분 추출
    // x축 기준으로 x축 범위 안에 있는 포인트들만 남김 (쭉 긴 직사각형)
    pcl::PassThrough <pcl::PointXYZI> x_filter; //필터 객체 생성

    x_filter.setInputCloud (voxel_cloud);
    x_filter.setFilterFieldName("x");
    x_filter.setFilterLimits(-car_size, car_size);
    x_filter.setFilterLimitsNegative(false); // car size 범위 내에 있는 포인트만 남김
    x_filter.filter(*x_inside);

    // x축 기준, car size 범위 밖에 있는 포인트 남김
    // 한 번 setInputCloud를 해두면 여러가지로 filter을 계속 할 수 있음
    x_filter.setFilterLimitsNegative(true);
    x_filter.filter(*x_outside);

    pcl::PassThrough <pcl::PointXYZI> y_filter;
    y_filter.setInputCloud (x_inside); // x축 범위 안에 있는 포인트 클라우드를 inputcloud로 받아온다. 
    y_filter.setFilterFieldName("y"); 
    y_filter.setFilterLimits(-car_size, car_size);
    y_filter.setFilterLimitsNegative(true); // x축 범위 안에서 y축 범위 안쪽에 있는 포인트 삭제
    y_filter.filter(*roi_cloud); // y축 범위만큼 삭제된 x축 포인트 클라우드 저장

    *roi_cloud += *x_outside; 

    //------------------------------------------------------------------------------------------

    // 3. ransac 

    // 찾은 평면이 뭐냐. 세그멘테이션으로 찾은 평면 방정식 계수들이 저장
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    // 감지하고자 하는 평면을 구성하는 모든 점들의 인덱스 리스트를 저장하는 데 사용되는 클래스
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); 

    // 세그먼테이션 객체 생성 (찾은 모델을 분리함)
    pcl::SACSegmentation<pcl::PointXYZI> seg;

    // 최종적으로 뽑힌 inlier를 기준으로 모델 계수를 한 번 더 보정할지
    seg.setOptimizeCoefficients(true);

    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    // SACMODEL_PERPENDICULAR_PLANE : 어떤 축에 수직인 평면만 찾도록 제한하는 모델

    // z축에 거의 수직인 평면만 찾게 함
    seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));   
    seg.setEpsAngle(10.0 * M_PI / 180.0);   


    seg.setMethodType(pcl::SAC_RANSAC);
    // 모델 찾는 방법 = 랜삭

    seg.setDistanceThreshold(0.3);  
    // 한 점에서 평면까지의 수직거리가 threshold 이하면 inlier로 포함
    // 0.3 으로 했을 땐 차체 기준 왼쪽 보도블럭? 까지 삭제

    // 최대 반복 횟수
    seg.setMaxIterations(200);

    // 입력 클라우드
    seg.setInputCloud(roi_cloud);
    // 세그먼테이션 수행 및 inliers, coefficients 결과 얻기
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0)
    {
        cout << "no model" << endl;
        return;
    }


    // -------------------------------------------------------------------------------------------

    // 4. 지면(inlier)과 비지면(outlier) 포인트 분리
    // 랜삭으로 지면을 찾고, 그 평면에 속한 점을 기준으로 포인트 클라우드를 분리하는 단계

    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZI>());
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_nonground(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr ransac_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    // cloud_nonground 를 ransac_cloud로 대체함

    pcl::ExtractIndices <pcl::PointXYZI> extract;
    // 평면을 구성하는 점들의 인덱스만 추출

    extract.setInputCloud(roi_cloud);
    extract.setIndices(inliers); 
    // 평면을 구성하는 점들의 인덱스

    // extract.setNegative(false); // false : Inlier (지면,지정된 인덱스) 추출
    // extract.filter(*cloud_ground);

    extract.setNegative(true); // true : Outlier (지면X 포인트) 추출
    extract.filter(*ransac_cloud);

    // ---------------------------------------------------------------------------------------------

    // pcl::PointCloud 를 다시 PointCloud2 형으로 변환
    sensor_msgs::PointCloud2 output_cloud;
    pcl::toROSMsg(*ransac_cloud, output_cloud); // 랜삭 결과는 지면X 포인트 출력
    output_cloud.header.frame_id = "lidar_link";
    // frame_id는 std_msgs::Header 메시지에 들어있는 정보. 
    output_cloud.header.stamp = input_cloud->header.stamp;

    pub_ransac.publish(output_cloud);

    // --------------------------------------------------------------------------------------------

    // 5. euclidean clustering

    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZI>);
    kdtree->setInputCloud(ransac_cloud);
    // -> : kdtree 포인터가 가리키는 객체 (힙에 생성되어 있는 KdTree) 의 멤버에 접근
    // ransac 결과 포인트 클라우드 (pcl::pointcloud)
    // (*kdtree).setInputCloud(ransac_cloud); 와 동일

    vector<pcl::PointIndices> cluster_indices;
    // PointIndices - 해당 클러스터를 구성하는 점들의 인덱스가 들어있음. 여기서 인덱스는 해당 점들이 전체 클라우드에서 몇 번째 위치에 있는지를 나타냄.
    // 벡터의 각 요소는 단일 클러스터에 속하는 포인트들의 인덱스 목록을 포함함. 

    // " 왜 포인트들을 복사해서 사용하지 않고 인덱스로 반환할까? "
    // -> 메모리, 시간 비용 크다
    // -> 얼마나 큰데? 
    
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> euclidean;
    // 클러스터링 알고리즘 구현 클래스

    // " 왜 kdtree 처럼 포인터가 아니라 객체를 생성했을까? "
    // -> KdTree는 여러 곳에서 공유해서 쓰는 반면, euclidean 은 클러스터링에서만 쓰는 알고리즘이기 때문에

    euclidean.setInputCloud(ransac_cloud);     
    euclidean.setSearchMethod(kdtree);               
    euclidean.setClusterTolerance(0.3); // 같은 클러스터로 묶일 수 있는 두 포인트 사이 최대 거리 (m)
    euclidean.setMinClusterSize(100);  // 최소 클러스터 크기 설정 (크기 = 점 개수) (int)
    euclidean.setMaxClusterSize(1000);       
    euclidean.extract(cluster_indices); // 알고리즘 실행 -> 클러스터 생성 (결과 : 포인트 인덱스 리스트)

    ROS_INFO_STREAM("---- clusters: " << cluster_indices.size() << " ----");

    int cid = 0;
    for (const auto& indices : cluster_indices) {
        int n = static_cast<int>(indices.indices.size());
        ROS_INFO_STREAM("cluster[" << cid << "] points = " << n);
        cid++;
    }

    //------------------------------------------------------------------------------------

    //  clustering 시각화 (동영 bounding box로 대체)
    // 1) cav 클러스터만 보이게. -> 클러스터 크기 정보를 알아야. detection 구조체에 저장되는 클러스터 정보 이용
    // 2) cav 클러스터에 bounding box로 시각화

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color(new pcl::PointCloud<pcl::PointXYZRGB>());
    color->is_dense = true;
    color->points.reserve(ransac_cloud->points.size());

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
            // idx는 ransac_cloud 기준 인덱스
            const auto& src = ransac_cloud->points[idx];

            pcl::PointXYZRGB p;
            p.x = src.x;
            p.y = src.y;
            p.z = src.z;    
            p.r = r; p.g = g; p.b = b;

            color->points.push_back(p);
        }

        cluster_id++;
    }

    color->width  = static_cast<uint32_t>(color->points.size());
    color->height = 1;

    // PointCloud2로 변환해서 publish
    sensor_msgs::PointCloud2 cluster_msg;
    pcl::toROSMsg(*color, cluster_msg);

    cluster_msg.header.frame_id = "lidar_link";
    cluster_msg.header.stamp = input_cloud->header.stamp;

    pub_cluster.publish(cluster_msg);
}

int main(int argc, char** argv)    
{
    init(argc, argv, "func_3D");
    
    NodeHandle nh;

    Subscriber sub = nh.subscribe("/lidar3D", 1, callback);

    pub_ransac = nh.advertise<sensor_msgs::PointCloud2>("/ransac", 1);
    pub_cluster = nh.advertise<sensor_msgs::PointCloud2>("/cluster", 1);
    // pub_cluster_all = nh.advertise<sensor_msgs::PointCloud2>("/cluster_all", 1);
    // pub_cluster_cav = nh.advertise<sensor_msgs::PointCloud2>("/cluster_cav", 1);

    spin();
    //ros::spinOnce()

    return 0;
}

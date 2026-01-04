#include "hybrid_function.hpp"
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <ros/package.h>
using namespace std;

void gpsCallback(const morai_msgs::GPSMessage::ConstPtr& msg) {
    double x, y, z;
    wgs84ToENU(msg->latitude, msg->longitude, msg->altitude,
               lat0, lon0, h0, x, y, z);
    ego.x = x;
    ego.y = y;
}

void egoCallback(const morai_msgs::EgoVehicleStatus::ConstPtr& msg){
    ego.vel = msg->velocity.x;
}

void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    ego.yaw = quaternionToYaw(msg->orientation.x,
                              msg->orientation.y,
                              msg->orientation.z,
                              msg->orientation.w);

}


void controlLoop(const ros::TimerEvent&) {


    closeWaypointsIdx(ego, ctrl.close_idx);
    getTargetwaypoint(ego, ctrl.close_idx, ctrl.target_idx, ctrl.ld); // close waypoint에서 ld 보다 먼 target waypoint잡기
    getMaxCurvature(ctrl.close_idx, ctrl.lookahead_idx, ego.max_curvature); // close_idx + lookahead_idx까지 최대 곡률 뽑아내는 함수
    getTargetSpeed(ego.max_curvature, ctrl.target_vel); // 코너 있으면 감속 아님 타겟 속도로
    getsteering(ego,ctrl); // 타겟 웨이포인트, 속도로 조향각계산
    computePID(ego.vel, ctrl.target_vel, ctrl.accel, ctrl.brake); //속도제어
    pubCmd(ctrl);

    ROS_INFO("| ego_V: %.2f | Acc: %.2f | Brk: %.2f | Steer: %.2f", 
             ego.vel, ctrl.accel, ctrl.brake, ctrl.steering);
             
    ROS_INFO("| Max_K: %.4f | Ld: %.1f |", 
             ego.max_curvature, ctrl.ld);
}



int main(int argc, char** argv) {
    ros::init(argc, argv, "hybrid");
    ros::NodeHandle nh;

    // Get ROS package path for file locations
    std::string pkg = ros::package::getPath("roscpp_morai");
    std::string yaml_file = pkg + "/config/yaml_hybrid.yaml";

    // Load YAML parameters first
    if (!load_Params(yaml_file)) {
        ROS_FATAL("Failed to load YAML parameters!");
        return -1;
    }
    
    // Load path and reference point
    if (!load_Path_ref()) {
        ROS_FATAL("Failed to load path and reference files!");
        return -1;
    }
    
    // Preprocess curvature
    preprocessCurvature();
    
    cmd_pub = nh.advertise<morai_msgs::CtrlCmd>("/ctrl_cmd_0", 1);
    ros::Subscriber gps_sub = nh.subscribe("/gps", 1, gpsCallback);
    ros::Subscriber imu_sub = nh.subscribe("/imu", 1, imuCallback);
    ros::Subscriber ego_sub = nh.subscribe("/Ego_topic", 1, egoCallback);
    ros::Timer timer = nh.createTimer(ros::Duration(0.02), controlLoop);

    ros::spin();

    return 0;
}


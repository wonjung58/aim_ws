#ifndef HYBRID_HPP
#define HYBRID_HPP

#include <ros/ros.h>
#include <morai_msgs/CtrlCmd.h>
#include <morai_msgs/GPSMessage.h>
#include <morai_msgs/EgoVehicleStatus.h>
#include <sensor_msgs/Imu.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

// ====================== 구조체 정의 ======================
struct Waypoint {
    double x, y;
    double curvature;
};

struct VehicleState {
    double x = 0.0, y = 0.0;
    double yaw = 0.0;
    double vel = 0.0;
    double max_curvature = 0.0;
};

struct ControlData {
    int close_idx = 0;
    int target_idx = 0;
    int lookahead_idx = 0;
    double target_vel = 0.0;
    double accel = 0.0;
    double brake = 0.0;
    double steering = 0.0;
    double ld = 0.0;
};

// ====================== 전역 변수 선언 (Extern) ======================
// (실제 메모리는 cpp 파일에 잡힘)
extern std::vector<Waypoint> waypoints;
extern VehicleState ego;
extern ControlData ctrl;
extern ros::Publisher cmd_pub;

// 야뮬 설정값 변수들
extern std::string path_file_name;
extern std::string ref_file_name;
extern double target_vel;       // 평소 속도
extern double curve_vel;        // 코너 속도
extern double curve_standard;   // 코너 판단 기준 곡률
extern double k_gain;           // Stanley Gain
extern double ld_gain;          // Pure Pursuit Gain
extern int lookahead_idx;       // 곡률 탐색 범위
extern double Kp, Ki, Kd;       // PID Gains

// 좌표계 기준점
extern double lat0, lon0, h0;
extern double x0_ecef, y0_ecef, z0_ecef;


// ====================== 함수 선언 ======================
// 좌표 변환
void wgs84ToENU(double lat, double lon, double h, double lat_ref, double lon_ref, double h_ref, double& x, double& y, double& z);
void wgs84ToECEF(double lat, double lon, double h, double& x, double& y, double& z);     
double quaternionToYaw(double x, double y, double z, double w);

// 초기화 및 유틸
bool load_Path_ref();
bool load_Params(const std::string& yaml_hybrid);
void preprocessCurvature();
void pubCmd(const ControlData& data);

// 제어 로직
void closeWaypointsIdx(const VehicleState& ego, int& out_idx);
void getTargetwaypoint(const VehicleState& ego, int close_idx, int& out_target_idx, double& ld);
void getMaxCurvature(int close_idx, int lookahead_idx, double& max_curvature);
void getTargetSpeed(double max_curvature, double& out_target_vel);

// PID 및 Steering (순서 중요함수들)
void computePID(double vel, double target_vel, double& out_accel, double& out_brake);
void getsteering(const VehicleState& ego, ControlData& ctrl);
double lateralPathError(int target_idx, double x, double y);
double headingError(double yaw, int target_idx);

#endif

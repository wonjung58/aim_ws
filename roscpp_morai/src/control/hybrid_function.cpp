#include "hybrid_function.hpp"
#include <morai_msgs/CtrlCmd.h>
#include <cmath>
#include <ros/ros.h>
#include <ros/package.h>
#include <yaml-cpp/yaml.h>
#include <iostream> 
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <utility>

using namespace std;

// ====================== 전역 변수 정의 ======================
std::vector<Waypoint> waypoints;
VehicleState ego;
ControlData ctrl;
ros::Publisher cmd_pub;

// YAML 변수
string path_file_name;
string ref_file_name;
double target_vel;
double curve_vel;
double curve_standard;
double k_gain;
double ld_gain;
int lookahead_idx;
double Kp, Ki, Kd;


// --- 차량/경로 상태 변수 ---
double lat0, lon0, h0;
double x0_ecef, y0_ecef, z0_ecef;



// ====================== 좌표 변환 함수 ======================
void wgs84ToECEF(double lat, double lon, double h,
                 double& x, double& y, double& z) {
    double a = 6378137.0;
    double e2 = 6.69437999014e-3;

    double rad_lat = lat * M_PI / 180.0;
    double rad_lon = lon * M_PI / 180.0;
    double N = a / sqrt(1 - e2 * sin(rad_lat) * sin(rad_lat));

    x = (N + h) * cos(rad_lat) * cos(rad_lon);
    y = (N + h) * cos(rad_lat) * sin(rad_lon);
    z = (N * (1 - e2) + h) * sin(rad_lat);
}

void wgs84ToENU(double lat, double lon, double h,
                double lat_ref, double lon_ref, double h_ref,
                double& x, double& y, double& z) {
    double x_ecef, y_ecef, z_ecef;
    wgs84ToECEF(lat, lon, h, x_ecef, y_ecef, z_ecef);

    double dx = x_ecef - x0_ecef;
    double dy = y_ecef - y0_ecef;
    double dz = z_ecef - z0_ecef;

    double rad_lat = lat_ref * M_PI / 180.0;
    double rad_lon = lon_ref * M_PI / 180.0;

    double t[3][3] = {
        {-sin(rad_lon), cos(rad_lon), 0},
        {-sin(rad_lat) * cos(rad_lon), -sin(rad_lat) * sin(rad_lon), cos(rad_lat)},
        {cos(rad_lat) * cos(rad_lon), cos(rad_lat) * sin(rad_lon), sin(rad_lat)}
    };

    x = t[0][0]*dx + t[0][1]*dy + t[0][2]*dz;
    y = t[1][0]*dx + t[1][1]*dy + t[1][2]*dz;
    z = t[2][0]*dx + t[2][1]*dy + t[2][2]*dz;
}

double quaternionToYaw(double x, double y, double z, double w) {
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return atan2(siny_cosp, cosy_cosp);
}

// ====================== 스탠리에 쓰이는 함수 ======================
double lateralPathError(int target_idx, double x, double y){ //정의
    int i = target_idx;
    if( i < waypoints.size() -1){
    double a = waypoints[i].x;
    double b = waypoints[i].y;
    double B = waypoints[i+1].x - waypoints[i].x;
    double A = waypoints[i+1].y - waypoints[i].y;
    double C = -a*A + b*B; // A*x -B*y + C;
    if(A==0 && B==0){return 0.0; }
    return (x*A - y*B+ C ) /sqrt(A*A + B*B);
    }
    else { return 0.0; }
}

double headingError(double yaw,int target_idx){
    int i = target_idx;
    if(i< waypoints.size()-1){
    double dx = waypoints[i+1].x - waypoints[i].x;
    double dy = waypoints[i+1].y - waypoints[i].y;
    double path_heading = atan2(dy,dx);
    double diff = path_heading - yaw;
    return atan2(sin(diff),cos(diff));
    }
    else {return 0.0;}
}

// ====================== main안 함수  ======================

bool load_Path_ref(){ //경로,기준점 체크 함수
    // Convert relative paths to absolute using ROS package path
    std::string pkg = ros::package::getPath("roscpp_morai");
    if (ref_file_name[0] != '/') {  
        ref_file_name = pkg + "/" + ref_file_name;
    }
    if (path_file_name[0] != '/') {
        path_file_name = pkg + "/" + path_file_name;
    }

    ROS_INFO("Opening ref file: %s", ref_file_name.c_str());
    ifstream ref_file(ref_file_name);
    if (!ref_file.is_open()) {
        ROS_ERROR("Failed to open ref file: %s", ref_file_name.c_str());
        return false;
    }
    ref_file >> lat0 >> lon0 >> h0;
    ref_file.close();
    wgs84ToECEF(lat0, lon0, h0, x0_ecef, y0_ecef, z0_ecef);

    ROS_INFO("Opening path file: %s", path_file_name.c_str());
    ifstream path_file(path_file_name);
    if (!path_file.is_open()) {
        ROS_ERROR("Failed to open path file: %s", path_file_name.c_str());
        return false;
    }

    // --- Parse CSV (space or comma separated) ---
    string line;
    while (getline(path_file, line)) {
        if (line.empty()) continue;
        
        stringstream ss(line);
        double x, y, z;
        
        // Try space-separated format first
        if (ss >> x >> y >> z) {
            waypoints.push_back({x, y});
        } else {
            // Try comma-separated format
            ss.clear();
            ss.str(line);
            string val;
            vector<double> row;
            
            while (getline(ss, val, ',')) {
                val.erase(0, val.find_first_not_of(" \t\r\n"));
                val.erase(val.find_last_not_of(" \t\r\n") + 1);
                
                try {
                    row.push_back(stod(val));
                } catch (...) {
                    continue;
                }
            }
            
            if (row.size() >= 2) {
                waypoints.push_back({row[0], row[1]});
            }
        }
    }
    // ------------------------------------

    path_file.close();
    ROS_INFO("Waypoints loaded: %zu", waypoints.size());

    // 데이터가 너무 적으면 곡률 계산 시 세그먼테이션 폴트가 발생하므로 체크
    if (waypoints.size() < 3) {
        ROS_ERROR("Not enough waypoints (size: %zu). Curvature calculation requires at least 3 points.", waypoints.size());
        return false;
    }

    return true;
}


bool load_Params(const std::string& yaml_hybrid) { //야뮬 체크 함수
  try {
    YAML::Node config3 = YAML::LoadFile(yaml_hybrid);

    // ==========================================================
    // 여기에 변환 상수를 추가합니다
    // ==========================================================
    const double KPH_TO_MPS = 1.0 / 3.6; // (km/h -> m/s 변환 상수)
    // ==========================================================

    // 1. 임시 변수 (km/h)에 원본 값 로드
    // (로그 출력을 위해 원본 km/h 값을 보관)
    double target_vel_kph     = config3["target_vel"].as<double>();
    double curve_vel_kph    = config3["curve_vel"].as<double>();



    // 2. 나머지 파라미터 로드 (단위 변환 불필요)
    curve_standard     = config3["curve_standard"].as<double>();
    k_gain           = config3["k_gain"].as<double>();
    ld_gain          = config3["ld_gain"].as<double>();
    lookahead_idx= config3["lookahead_idx"].as<int>();
    Kp               = config3["Kp"].as<double>();
    Ki               = config3["Ki"].as<double>();
    Kd               = config3["Kd"].as<double>();
    path_file_name   = config3["path_file_name"].as<std::string>();
    ref_file_name    = config3["ref_file_name"].as<std::string>();

    
    // ==========================================================
    // 3. (중요) 전역 변수에 m/s로 변환하여 저장
    // ==========================================================
    target_vel = target_vel_kph * KPH_TO_MPS;
    curve_vel = curve_vel_kph * KPH_TO_MPS;



    // 4. 로그 출력 (사용자 가독성을 위해 km/h 원본 값과 m/s 변환 값 함께 표시)
    cout << "\n======================================\n";
    cout << "Hybrid Controller Parameters (Loaded)\n";
    cout << "======================================\n";
    
    cout << "--- 1. Path Files ---\n";
    cout << "path_file_name:   " << path_file_name << endl;
    cout << "ref_file_name:    " << ref_file_name << endl;
    
    cout << "\n--- 2. FSM & Speed Planning (Converted to m/s) ---\n";
    cout << "target_vel:       " << target_vel_kph << " (km/h) -> " << target_vel << " (m/s)" << endl;
    cout << "curve_vel:  " << curve_vel_kph <<  " (km/h) -> " << curve_vel << " (m/s)" << endl;
    cout << "lookahead_idx:  " << lookahead_idx << " (indices)" << endl;
    cout << "curve_standard:      " << curve_standard << " (curvature)" << endl;

    cout << "\n--- 3. PID Control (Velocity) ---\n";
    cout << "Kp:               " << Kp << endl;
    cout << "Ki:               " << Ki << endl;
    cout << "Kd:               " << Kd << endl;

    cout << "\n--- 4. Steering Gains ---\n";
    cout << "k_gain (Stanley): " << k_gain << endl;
    cout << "ld_gain (PP):     " << ld_gain << endl;


    cout << "======================================\n" << endl;
    ctrl.lookahead_idx = lookahead_idx;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "YAML Load Error: " << e.what() << std::endl;
    return false;
  }
}

void preprocessCurvature() {
    for (int i=0; i < waypoints.size()-2; ++i){
        double dx1 = waypoints[i+1].x - waypoints[i].x;
        double dx2 = waypoints[i+2].x - waypoints[i+1].x;
        double dy1 = waypoints[i+1].y - waypoints[i].y;
        double dy2 = waypoints[i+2].y - waypoints[i+1].y;
        double alpha1 = std::atan2(dy1, dx1);
        double alpha2 = std::atan2(dy2, dx2);
        double k_val = fabs(alpha1 - alpha2);
        waypoints[i+1].curvature = k_val;
        // ← i+1번 점에 저장!
    }
    cout << "=== End Preprocessing ===" << endl;
    waypoints[0].curvature = waypoints[1].curvature;  // ← 보간
    waypoints.back().curvature = waypoints[waypoints.size()-2].curvature;
}


// ====================== 출발/도착 판정 함수 ======================



void closeWaypointsIdx(const VehicleState& ego, int& out_idx){ //const는 구조체 전체를 읽기 전용으로 받음 ,출력: int& out_idx -> 결과를 여기에 담으면 ctrl.close_idx가 바뀜
    static int last_close_idx = 0;
    double best_close_dist = 10000000000;
    int close_idx = last_close_idx;
    int start = std::max(0,last_close_idx - 10);
    int end = std::min((int)waypoints.size() - 1,last_close_idx + 30);
    for(int i = start; i <= end ; ++i){
        double dx = waypoints[i].x - ego.x;
        double dy = waypoints[i].y - ego.y;
        double dist = sqrt(dx*dx + dy*dy);
        if (dist < best_close_dist){
                best_close_dist = dist;
                close_idx = i;
            }
           
        }
    last_close_idx = close_idx;
    out_idx = close_idx;
    ROS_INFO("close_idx: %d",close_idx);
}

void getTargetwaypoint(const VehicleState& ego, int close_idx, int& out_target_idx, double& ld){
    ld = 5.0 + ld_gain * ego.vel;
    int target_idx = close_idx;
    int i = close_idx;
    for(; i <= waypoints.size()-1; ++i ){
        double dx = waypoints[i].x-ego.x;
        double dy = waypoints[i].y-ego.y;
        double dist = sqrt(dx*dx + dy*dy);
        if(dist > ld){
            target_idx = i;
            break;
          }
        }
    out_target_idx = target_idx;
}

void getMaxCurvature(int close_idx, int lookahead_idx, double& max_curvature){
    
    double max_kappa = 0.0;
    int end_idx = min((int)waypoints.size(), close_idx + lookahead_idx);
    // close_idx부터 end_idx 전까지만 탐색
    for (int i = close_idx; i < end_idx; ++i) {
        double now_kappa = waypoints[i].curvature; 
        if (now_kappa > max_kappa) {
            max_kappa = now_kappa;
        }
    } 
    max_curvature = max_kappa;

}

void getTargetSpeed(double max_curvature, double& out_target_vel){
    if(max_curvature > curve_standard){
        out_target_vel = curve_vel;
    }
    else {out_target_vel = target_vel;}
}

void getsteering(const VehicleState& ego, ControlData& ctrl){
    double path_e = lateralPathError(ctrl.target_idx, ego.x, ego.y);
    double heading_e = headingError(ego.yaw, ctrl.target_idx);
    double v = std::max(1.0, ego.vel);
    
    // Stanley steering control with limits
    double steering_raw = heading_e + atan(k_gain * path_e / v);
    
    // Limit steering angle to ±30 degrees (±0.524 radians)
    const double MAX_STEERING = 30.0 * M_PI / 180.0;  // 30 degrees
    ctrl.steering = std::max(-MAX_STEERING, std::min(MAX_STEERING, steering_raw));
    
    ROS_INFO("[Steering] path_e: %.3f, heading_e: %.3f, raw: %.3f, limited: %.3f deg",
             path_e, heading_e, steering_raw, ctrl.steering * 180.0 / M_PI);
}

void computePID(double vel,double target_vel,double& out_accel, double& out_brake){

    static double prev_error = 0.0;     // 이전 오차 기억용
    static double integral_error = 0.0; // 적분 누적용
    double error = target_vel - vel;
    integral_error += error * 0.02;

    if(integral_error > 10.0) integral_error = 10.0;
    if(integral_error < -10.0) integral_error = -10.0;

    double p_error = Kp*error;
    double i_error = Ki*integral_error; //controlLoop가 0.02초(50Hz)마다 도니까 dt = 0.02
    double d_error = Kd*((error - prev_error)/0.02);
    prev_error = error;

    double total_output = p_error + i_error + d_error;

    if (total_output > 0) {
        out_accel = min(total_output, 1.0);
        out_brake = 0.0;
    } 
    else {
        out_accel = 0.0;
        out_brake = min(-total_output, 1.0); 
    }

}


void pubCmd(const ControlData& data) {
    morai_msgs::CtrlCmd cmd;
    cmd.longlCmdType = 1;
    cmd.accel = data.accel;
    cmd.brake = data.brake;
    cmd.steering = data.steering;
    cmd_pub.publish(cmd);
}




#include <ros/ros.h>
#include <morai_msgs/GPSMessage.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// 기준점
double lat0, lon0, h0;
double x0_ecef, y0_ecef, z0_ecef;

struct Point {
    double x, y, z;  // ENU 좌표
};

Point curr_pos;
Point last_saved_pos = {0.0, 0.0, 0.0}; 
bool is_gps = false; 
bool first_record = true;

// WGS84 -> ECEF 변환
void wgs84ToECEF(double lat, double lon, double h, double& x, double& y, double& z) {
    double a = 6378137.0;
    double e2 = 6.69437999014e-3;
    double rad_lat = lat * M_PI / 180.0;
    double rad_lon = lon * M_PI / 180.0;
    double N = a / sqrt(1 - e2 * sin(rad_lat) * sin(rad_lat));
    x = (N + h) * cos(rad_lat) * cos(rad_lon);
    y = (N + h) * cos(rad_lat) * sin(rad_lon);
    z = (N * (1 - e2) + h) * sin(rad_lat);
}

// WGS84 -> ENU 변환
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

void gpsCallback(const morai_msgs::GPSMessage::ConstPtr& msg) {
    // GPS를 ENU로 변환
    wgs84ToENU(msg->latitude, msg->longitude, msg->altitude,
               lat0, lon0, h0, 
               curr_pos.x, curr_pos.y, curr_pos.z);
    is_gps = true;
}

double calculateDistance(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "path_recorder");
    ros::NodeHandle nh;

    // ref.txt 읽기
    string ref_path = "/root/aim_ws/src/roscpp_morai/config/ref.txt";
    ifstream ref_file(ref_path);
    if (!ref_file.is_open()) {
        ROS_ERROR("Failed to open ref file: %s", ref_path.c_str());
        return -1;
    }
    ref_file >> lat0 >> lon0 >> h0;
    ref_file.close();
    
    // 기준점 ECEF 변환
    wgs84ToECEF(lat0, lon0, h0, x0_ecef, y0_ecef, z0_ecef);
    
    ROS_INFO("Reference point loaded: lat=%.8f, lon=%.8f", lat0, lon0);

    ros::Subscriber gps_sub = nh.subscribe("/gps", 10, gpsCallback);

    string file_path = "/root/aim_ws/src/roscpp_morai/config/track_log_recorded.csv";
    ofstream out_file(file_path); 

    if (!out_file.is_open()) {
        ROS_ERROR("Failed to open file: %s", file_path.c_str());
        return -1;
    }

    ROS_INFO("Path Recorder Started - Saving ENU coordinates");
    ROS_INFO("Saving to: %s", file_path.c_str());

    ros::Rate rate(50); 
    int log_count = 0;

    while (ros::ok()) {
        ros::spinOnce();

        if (is_gps) {
            double dist = 0.0;
            if (!first_record) {
                dist = calculateDistance(curr_pos, last_saved_pos);
            }

            if (dist >= 0.5 || first_record) {
                // ENU 좌표 저장 (쉼표로 구분)
                out_file << fixed << setprecision(6) 
                         << curr_pos.x << "," 
                         << curr_pos.y << "," 
                         << curr_pos.z << "\n";
                out_file.flush();
                
                ROS_INFO("Recorded ENU: x=%.2f y=%.2f z=%.2f (dist=%.2f m)", 
                         curr_pos.x, curr_pos.y, curr_pos.z, dist);
                
                last_saved_pos = curr_pos;
                first_record = false;
            }

            if (++log_count >= 100) {
                ROS_INFO("Recording... Current ENU: (%.2f, %.2f, %.2f)", curr_pos.x, curr_pos.y, curr_pos.z);
                log_count = 0;
            }
        } else {
            if (++log_count >= 100) {
                ROS_WARN("Waiting for /gps message...");
                log_count = 0;
            }
        }

        rate.sleep();
    }

    out_file.close();
    return 0;
}
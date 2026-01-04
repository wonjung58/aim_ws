#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/UInt8.h>
#include <morai_msgs/CtrlCmd.h>
#include <morai_msgs/EgoVehicleStatus.h>
#include <morai_msgs/GPSMessage.h>
#include <tf/transform_datatypes.h>

#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geocentric.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>

// ======================== 전역 상태 ========================

// pubs/subs
ros::Publisher  cmd_pub_;
ros::Publisher  path_pub_;

// 메시지
nav_msgs::Path      path_msg_;
morai_msgs::CtrlCmd cmd_;

// ENU origin, 상태
GeographicLib::LocalCartesian lc_(GeographicLib::Geocentric::WGS84());
bool have_origin_ = false;
bool have_gps_    = false;
bool have_yaw_    = false;

double enu_x_ = 0.0;
double enu_y_ = 0.0;
double yaw_   = 0.0;
double ego_speed_ms_ = 0.0;

// path
std::vector<std::pair<double,double>> path_xy_;
std::string ref_file_;
std::string path_file_;
std::string flag_topic_;

// control params
double wheelbase_L    = 3.0;   // [m]
double lfd_           = 4.5;   // [m]
double target_vel_kmh = 15.0;  // [km/h]
double avoid_kmh_     = 8.0;   // [km/h]

// 회피 모드
enum class Mode { TRACK_WAYPOINT, AVOID_LEFT, AVOID_RIGHT };
Mode mode_ = Mode::TRACK_WAYPOINT;
bool avoid_first_step_hard_right_ = false;


// ======================== 유틸/파일 로드 ========================

bool loadOrigin(const std::string &file) {
  std::ifstream in(file);
  if (!in.is_open()) {
    ROS_ERROR("[pp_fixed] Failed to open origin file: %s", file.c_str());
    return false;
  }
  double lat0, lon0, alt0;
  in >> lat0 >> lon0 >> alt0;
  lc_.Reset(lat0, lon0, alt0);
  have_origin_ = true;
  ROS_INFO("[pp_fixed] ENU origin: lat=%.15f lon=%.15f alt=%.3f",
           lat0, lon0, alt0);
  return true;
}

bool loadPath(const std::string &file) {
  std::ifstream in(file);
  if (!in.is_open()) {
    ROS_ERROR("[pp_fixed] Failed to open path file: %s", file.c_str());
    return false;
  }
  path_xy_.clear();
  std::string line;
  double x, y, z;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    if (!(iss >> x >> y >> z)) continue;
    path_xy_.emplace_back(x, y);
  }
  ROS_INFO("[pp_fixed] Path points loaded: %zu", path_xy_.size());
  return !path_xy_.empty();
}

void publishPath() {
  nav_msgs::Path p;
  p.header.stamp = ros::Time::now();
  p.header.frame_id = "map";

  for (auto &pt : path_xy_) {
    geometry_msgs::PoseStamped ps;
    ps.header = p.header;
    ps.pose.position.x = pt.first;
    ps.pose.position.y = pt.second;
    ps.pose.orientation.w = 1.0;
    p.poses.push_back(ps);
  }
  path_pub_.publish(p);
  path_msg_ = p;
}

int findNearestIdx(double x, double y) {
  if (path_xy_.empty()) return -1;
  int best_idx = -1;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < path_xy_.size(); ++i) {
    double dx = path_xy_[i].first  - x;
    double dy = path_xy_[i].second - y;
    double d2 = dx*dx + dy*dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

void publishStop() {
  cmd_.steering = 0.0;
  cmd_.velocity = 0.0;
  cmd_.accel    = 0.0;
  cmd_.brake    = 0.0;
  cmd_pub_.publish(cmd_);
}


// ======================== 콜백 ========================

void gpsCB(const morai_msgs::GPSMessage &msg) {
  if (!have_origin_) {
    ROS_WARN_THROTTLE(1.0,
      "[pp_fixed] gpsCB called but origin not loaded yet");
    return;
  }
  double x, y, z;
  try {
    lc_.Forward(msg.latitude, msg.longitude,
                std::isfinite(msg.altitude) ? msg.altitude : 0.0,
                x, y, z);
  } catch (...) {
    ROS_WARN_THROTTLE(1.0, "[pp_fixed] LocalCartesian.Forward failed");
    return;
  }
  enu_x_ = x;
  enu_y_ = y;
  have_gps_ = true;
}

void imuCB(const sensor_msgs::Imu &msg) {
  const auto &q = msg.orientation;
  tf::Quaternion tfq(q.x, q.y, q.z, q.w);
  yaw_ = tf::getYaw(tfq);
  have_yaw_ = true;
}

void egoCB(const morai_msgs::EgoVehicleStatus &msg) {
  ego_speed_ms_ = msg.velocity.x;
}

void flagCB(const std_msgs::UInt8::ConstPtr& m) {
  const uint8_t f = m->data;
  switch (f) {
    case 0:
      mode_ = Mode::TRACK_WAYPOINT;
      avoid_first_step_hard_right_ = false;
      break;
    case 1:
      mode_ = Mode::AVOID_LEFT;
      break;
    case 2:
      mode_ = Mode::AVOID_RIGHT;
      break;
    case 3:
      mode_ = Mode::AVOID_RIGHT;
      avoid_first_step_hard_right_ = true;
      break;
    default:
      break;
  }
}


// ======================== 메인 루프 ========================

int main(int argc, char **argv) {
  ros::init(argc, argv, "pp_fixed_speed");

  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // ---- 파라미터 ----
  pnh.param<std::string>("ref_file",   ref_file_,
                         std::string("/root/ws/src/roscpp_morai/map/ref.txt"));
  pnh.param<std::string>("path_file",  path_file_,
                         std::string("/root/ws/src/roscpp_morai/map/Path.txt"));
  pnh.param<double>("wheelbase",       wheelbase_L,    3.0);
  pnh.param<double>("lookahead",       lfd_,           4.5);
  pnh.param<double>("target_kmh",      target_vel_kmh, 15.0);
  pnh.param<double>("avoid_kmh",       avoid_kmh_,     8.0);
  pnh.param<std::string>("flag_topic", flag_topic_,
                         std::string("/avoid_flag"));

  // ---- ENU 원점/경로 로드 ----
  if (!loadOrigin(ref_file_)) {
    ROS_FATAL("[pp_fixed] Failed to load ENU origin from %s",
              ref_file_.c_str());
    return 1;
  }
  if (!loadPath(path_file_)) {
    ROS_FATAL("[pp_fixed] Failed to load path from %s",
              path_file_.c_str());
    return 1;
  }

  // ---- PUB/SUB ----
  path_pub_ = nh.advertise<nav_msgs::Path>("/local_path", 1, true);
  publishPath();

  ros::Subscriber gps_sub  = nh.subscribe("/gps",       1, gpsCB);
  ros::Subscriber imu_sub  = nh.subscribe("/imu",       1, imuCB);
  ros::Subscriber ego_sub  = nh.subscribe("/Ego_topic", 1, egoCB);
  ros::Subscriber flag_sub = nh.subscribe<std_msgs::UInt8>(flag_topic_, 10,
                                                           flagCB);

  cmd_pub_ = nh.advertise<morai_msgs::CtrlCmd>("/ctrl_cmd", 2);

  // 제어 모드: velocity 제어
  cmd_.longlCmdType = 2;
  cmd_.accel  = 0.0;
  cmd_.brake  = 0.0;

  mode_ = Mode::TRACK_WAYPOINT;
  avoid_first_step_hard_right_ = false;

  ros::Rate rate(50);
  while (ros::ok()) {
    ros::spinOnce();

    // GPS, YAW 대기
    if (!have_gps_) {
      ROS_INFO_THROTTLE(1.0, "[pp_fixed] waiting /gps ...");
      rate.sleep();
      continue;
    }
    if (!have_yaw_) {
      ROS_INFO_THROTTLE(1.0, "[pp_fixed] waiting /imu/data (yaw) ...");
      rate.sleep();
      continue;
    }

    // ================= A. 회피 모드 =================
    if (mode_ == Mode::AVOID_LEFT || mode_ == Mode::AVOID_RIGHT) {
      if (avoid_first_step_hard_right_) {
        cmd_.steering = 1.0;   // 첫 스텝 하드 우방향 회피
        avoid_first_step_hard_right_ = false;
      } else {
        if (mode_ == Mode::AVOID_RIGHT) cmd_.steering = 0.8;
        else                            cmd_.steering = 0.2;
      }
      cmd_.velocity = avoid_kmh_;
      cmd_.accel = 0.0;
      cmd_.brake = 0.0;
      cmd_pub_.publish(cmd_);

      const int nearest_idx_tmp = findNearestIdx(enu_x_, enu_y_);
      ROS_INFO_THROTTLE(0.5,
        "[pp_fixed] AVOID ENU(%.2f, %.2f) yaw=%.1fdeg steer=%.3frad v_set=%.2f(km/h) (near=%d)",
        enu_x_, enu_y_, yaw_ * 180.0 / M_PI,
        cmd_.steering, cmd_.velocity, nearest_idx_tmp);

      rate.sleep();
      continue;
    }

    // ================= B. Pure Pursuit 모드 =================
    const int nearest_idx = findNearestIdx(enu_x_, enu_y_);
    if (nearest_idx < 0) {
      ROS_WARN_THROTTLE(1.0, "[pp_fixed] nearest index not found");
      publishStop();
      rate.sleep();
      continue;
    }

    double lx = 0.0, ly = 0.0;
    bool found = false;

    {
      const double cosv = std::cos(yaw_);
      const double sinv = std::sin(yaw_);
      const int path_size = static_cast<int>(path_xy_.size());

      for (int i = nearest_idx; i < path_size; ++i) {
        const double dx = path_xy_[i].first  - enu_x_;
        const double dy = path_xy_[i].second - enu_y_;
        const double x_local =  cosv * dx + sinv * dy;  // 전방(+)
        const double y_local = -sinv * dx + cosv * dy;  // 좌(+)

        if (x_local > 0.0) {
          const double d = std::hypot(x_local, y_local);
          if (d >= lfd_) {
            lx = x_local;
            ly = y_local;
            found = true;
            break;
          }
        }
      }

      if (!found) {
        const double dx = path_xy_.back().first  - enu_x_;
        const double dy = path_xy_.back().second - enu_y_;
        lx =  cosv * dx + sinv * dy;
        ly = -sinv * dx + cosv * dy;
        found = (lx > 0.0);
      }
    }

    if (!found) {
      ROS_WARN_THROTTLE(1.0,
        "[pp_fixed] forward point not found (lfd=%.1f)", lfd_);
      publishStop();
      rate.sleep();
      continue;
    }

    const double theta = std::atan2(ly, lx);
    const double delta = std::atan2(2.0 * wheelbase_L * std::sin(theta), lfd_);
    cmd_.steering = delta;           // [rad]
    cmd_.velocity = target_vel_kmh;  // [km/h]

    cmd_.accel = 0.0;
    cmd_.brake = 0.0;
    cmd_pub_.publish(cmd_);

    ROS_INFO_THROTTLE(0.5,
      "[pp_fixed] TRACK ENU(%.2f, %.2f) yaw=%.1fdeg steer=%.3frad v_set=%.2f(km/h) (near=%d)",
      enu_x_, enu_y_, yaw_ * 180.0 / M_PI,
      cmd_.steering, cmd_.velocity, nearest_idx);

    rate.sleep();
  }

  return 0;
}

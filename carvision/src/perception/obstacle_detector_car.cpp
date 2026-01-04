#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <XmlRpcValue.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

struct Detection {
  cv::Rect box;
  int class_id;
  float score;
};

static ros::Publisher pub_bbox_info;
static ros::Publisher pub_bbox_x;
static ros::Publisher pub_bbox_height;
static cv::dnn::Net   yolo_net;
static bool           yolo_ready = false;

static std::set<int> target_class_ids;
static std::string   model_path;
static std::string   camera_topic;
static double        avoid_height_thresh = 30.0;
static double        avoid_x_thresh      = 350.0;
static double        conf_threshold      = 0.3;
static double        nms_threshold       = 0.45;
static int           input_width         = 640;
static int           input_height        = 640;

bool parseTargetClasses(ros::NodeHandle& nh, const std::string& param_name,
                        std::set<int>& out) {
  XmlRpc::XmlRpcValue list;
  if (!nh.getParam(param_name, list)) {
    return false;
  }
  if (list.getType() != XmlRpc::XmlRpcValue::TypeArray) {
    ROS_WARN("~%s must be an array. Ignoring.", param_name.c_str());
    return false;
  }
  std::set<int> result;
  for (int i = 0; i < list.size(); ++i) {
    if (list[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
      result.insert(static_cast<int>(list[i]));
    } else {
      ROS_WARN("~%s[%d] is not int. Skipping.", param_name.c_str(), i);
    }
  }
  if (!result.empty()) {
    out = result;
    return true;
  }
  return false;
}

bool loadModel(const std::string& path) {
  try {
    yolo_net = cv::dnn::readNet(path);
    yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  } catch (const cv::Exception& e) {
    ROS_ERROR("Failed to load YOLO model from %s: %s", path.c_str(), e.what());
    return false;
  }
  return true;
}

inline float getValue(const float* data, bool dim_first,
                      int dims, int proposals,
                      int dim_idx, int proposal_idx) {
  if (dim_first) {
    return data[dim_idx * proposals + proposal_idx];
  }
  return data[proposal_idx * dims + dim_idx];
}

std::vector<Detection> runInference(const cv::Mat& frame) {
  std::vector<Detection> detections;
  if (!yolo_ready) {
    return detections;
  }

  cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                                        cv::Size(input_width, input_height),
                                        cv::Scalar(), true, false);
  yolo_net.setInput(blob);
  std::vector<cv::Mat> outputs;
  yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames());
  if (outputs.empty()) {
    return detections;
  }

  const cv::Mat& output = outputs.front();
  if (output.dims < 3) {
    ROS_WARN_THROTTLE(1.0, "Unexpected YOLO output dims=%d", output.dims);
    return detections;
  }

  const int axis1 = output.size[1];
  const int axis2 = output.size[2];
  const bool dim_first = axis1 <= axis2;
  const int dims = dim_first ? axis1 : axis2;
  const int proposals = dim_first ? axis2 : axis1;
  if (dims < 6) {
    ROS_WARN_THROTTLE(1.0, "YOLO output dims too small (%d)", dims);
    return detections;
  }

  const float scale_x = static_cast<float>(frame.cols) / static_cast<float>(input_width);
  const float scale_y = static_cast<float>(frame.rows) / static_cast<float>(input_height);
  const float* data = reinterpret_cast<const float*>(output.data);

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;
  boxes.reserve(proposals);
  confidences.reserve(proposals);
  class_ids.reserve(proposals);

  for (int i = 0; i < proposals; ++i) {
    const float x_center = getValue(data, dim_first, dims, proposals, 0, i);
    const float y_center = getValue(data, dim_first, dims, proposals, 1, i);
    const float box_w    = getValue(data, dim_first, dims, proposals, 2, i);
    const float box_h    = getValue(data, dim_first, dims, proposals, 3, i);

    int best_class = -1;
    float best_score = 0.0f;
    for (int c = 4; c < dims; ++c) {
      const float score = getValue(data, dim_first, dims, proposals, c, i);
      if (score > best_score) {
        best_score = score;
        best_class = c - 4;
      }
    }

    if (best_class < 0 || best_score < conf_threshold) {
      continue;
    }
    if (!target_class_ids.empty() &&
        target_class_ids.find(best_class) == target_class_ids.end()) {
      continue;
    }

    float x = (x_center - box_w * 0.5f) * scale_x;
    float y = (y_center - box_h * 0.5f) * scale_y;
    float w = box_w * scale_x;
    float h = box_h * scale_y;
    x = std::max(0.0f, x);
    y = std::max(0.0f, y);
    w = std::min(w, static_cast<float>(frame.cols) - x);
    h = std::min(h, static_cast<float>(frame.rows) - y);

    boxes.emplace_back(static_cast<int>(x), static_cast<int>(y),
                       static_cast<int>(w), static_cast<int>(h));
    confidences.push_back(best_score);
    class_ids.push_back(best_class);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold,
                    nms_threshold, indices);
  detections.reserve(indices.size());
  for (int idx : indices) {
    Detection det;
    det.box = boxes[idx];
    det.class_id = class_ids[idx];
    det.score = confidences[idx];
    detections.push_back(det);
  }
  return detections;
}

void publishDefaultBBox() {
  std_msgs::Float32 msg_x;
  std_msgs::Float32 msg_h;
  msg_x.data = -1.0f;
  msg_h.data = 0.0f;
  pub_bbox_x.publish(msg_x);
  pub_bbox_height.publish(msg_h);
}

void publishMainBBox(double cx, double height_ratio) {
  std_msgs::Float32 msg_x;
  std_msgs::Float32 msg_h;
  msg_x.data = static_cast<float>(cx);
  msg_h.data = static_cast<float>(height_ratio);
  pub_bbox_x.publish(msg_x);
  pub_bbox_height.publish(msg_h);
}

void imageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
  cv::Mat raw_data(1, static_cast<int>(msg->data.size()), CV_8UC1,
                   const_cast<uint8_t*>(msg->data.data()));
  cv::Mat frame = cv::imdecode(raw_data, cv::IMREAD_COLOR);
  if (frame.empty()) {
    ROS_WARN_THROTTLE(1.0, "Failed to decode camera frame");
    publishDefaultBBox();
    return;
  }

  std::vector<Detection> detections = runInference(frame);

  double main_height_ratio = -1.0;
  double main_cx = -1.0;

  for (const auto& det : detections) {
    const double cx = det.box.x + det.box.width * 0.5;
    const double cy = det.box.y + det.box.height * 0.5;
    const double height_ratio =
        (det.box.height / static_cast<double>(frame.rows)) * 100.0;

    std_msgs::Float32MultiArray msg_out;
    msg_out.data = {
        static_cast<float>(cx),
        static_cast<float>(cy),
        static_cast<float>(height_ratio),
        static_cast<float>(det.class_id)
    };
    pub_bbox_info.publish(msg_out);

    if (height_ratio > main_height_ratio) {
      main_height_ratio = height_ratio;
      main_cx = cx;
    }
  }

  if (main_height_ratio >= 0.0) {
    publishMainBBox(main_cx, main_height_ratio);
  } else {
    publishDefaultBBox();
  }

  if (main_height_ratio >= avoid_height_thresh &&
      main_cx > avoid_x_thresh) {
    ROS_INFO_THROTTLE(1.0,
                      "[yolo_cpp] AVOID TRIGGER x=%.1f h=%.1f%% (thresh %.1f/%.1f)",
                      main_cx, main_height_ratio,
                      avoid_x_thresh, avoid_height_thresh);
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "obstacle_detector_car_cpp");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  pnh.param<std::string>("camera_topic", camera_topic,
                         std::string("/image_jpeg/compressed"));
  pnh.param<std::string>("model_path", model_path,
                         std::string("yolov8n.pt"));
  pnh.param<double>("avoid_height_thresh", avoid_height_thresh, 30.0);
  pnh.param<double>("avoid_x_thresh",      avoid_x_thresh,      350.0);
  pnh.param<double>("confidence_thresh",   conf_threshold,      0.3);
  pnh.param<double>("nms_threshold",       nms_threshold,       0.45);
  pnh.param<int>("input_width",            input_width,         640);
  pnh.param<int>("input_height",           input_height,        640);

  if (!parseTargetClasses(pnh, "target_class_ids", target_class_ids)) {
    target_class_ids = {2};
  }

  ROS_INFO("Loading YOLO model: %s", model_path.c_str());
  yolo_ready = loadModel(model_path);
  if (!yolo_ready) {
    ROS_ERROR("YOLO model failed to load. Exiting.");
    return 1;
  }

  pub_bbox_info = nh.advertise<std_msgs::Float32MultiArray>("/yolo/bbox_info", 10);
  pub_bbox_x    = nh.advertise<std_msgs::Float32>("/yolo/bbox_x", 10);
  pub_bbox_height = nh.advertise<std_msgs::Float32>("/yolo/bbox_height_ratio", 10);

  ros::Subscriber sub_img = nh.subscribe(camera_topic, 1, imageCallback);

  if (target_class_ids.empty()) {
    ROS_INFO("[yolo_cpp] target class ids: ALL");
  } else {
    std::string cls_str;
    for (int cls : target_class_ids) {
      cls_str += std::to_string(cls) + " ";
    }
    ROS_INFO("[yolo_cpp] target class ids: %s", cls_str.c_str());
  }

  ros::spin();
  return 0;
}

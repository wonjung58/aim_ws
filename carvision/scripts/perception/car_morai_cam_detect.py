#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Float32
from cv_bridge import CvBridge
from ultralytics import YOLO
import time


class YoloLaneDetectorNode:
    def __init__(self):
        rospy.init_node("yolo_lane_detector", anonymous=False)
        rospy.loginfo("[YOLO] Node started (bbox + class + center + height_ratio + topics)")

        # ROS parameters
        self.cam_topic = rospy.get_param("~camera_topic", "/image_jpeg/compressed")
        self.model_path = rospy.get_param("~model_path", "yolov8n.pt")

        # 회피 기준도 파라미터로 뺌 (기본: 높이 30%, x > 350)
        self.avoid_height_thresh = rospy.get_param("~avoid_height_thresh", 28.0)
        self.avoid_x_thresh      = rospy.get_param("~avoid_x_thresh", 330.0)
        # YOLO 클래스 필터 (COCO에서 'car'는 2). 빈 리스트면 전체 사용.
        target_cls_default = [2]
        self.target_class_ids = set(rospy.get_param("~target_class_ids", target_cls_default))
        rospy.loginfo("[YOLO] target classes: %s",
                      sorted(self.target_class_ids) if self.target_class_ids else "ALL")

        self.bridge = CvBridge()
        rospy.loginfo("[YOLO] loading model: %s", self.model_path)
        self.model = YOLO(self.model_path)

        rospy.Subscriber(self.cam_topic, CompressedImage, self.callback, queue_size=1)

        # 기존: bbox 전체 정보 퍼블리시 (cx, cy, h_ratio, class_id)
        self.pub_info = rospy.Publisher("/yolo/bbox_info", Float32MultiArray, queue_size=1)

        # 추가: x값, 높이 퍼센트를 각각 퍼블리시
        self.pub_x = rospy.Publisher("/yolo/bbox_x", Float32, queue_size=1)
        self.pub_height = rospy.Publisher("/yolo/bbox_height_ratio", Float32, queue_size=1)

        self.prev_time = time.time()

    # =============================
    #       YOLO Callback
    # =============================
    def callback(self, msg):
        # CompressedImage -> OpenCV BGR
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        h, w = frame.shape[:2]

        # YOLO detection
        results = self.model(frame, verbose=False)[0]

        # 시각화용 이미지
        vis = frame.copy()

        # 가장 "큰" 박스(높이 비율 최대)를 따로 저장해서 그걸 토픽으로 쏨
        main_cx = None
        main_h_ratio = None

        # ---------------------------------------------------------
        # YOLO bounding box loop
        # ---------------------------------------------------------
        for box in results.boxes:
            # bbox 좌표
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 프레임 높이 대비 바운딩박스 높이 비율 (%)
            height_ratio = ((y2 - y1) / float(h)) * 100.0

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 필요 클래스만 통과시켜서 엉뚱한 물체로 회피가 켜지는 걸 방지
            if self.target_class_ids and cls_id not in self.target_class_ids:
                continue

            # ======== 시각화 =========
            # 바운딩 박스
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 중심점
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

            # 라벨 (클래스 + 중심좌표 + 높이 퍼센트)
            # 예: ID:0 (0.92) C=(320,240) H=35.7%
            label = f"ID:{cls_id} ({conf:.2f}) C=({int(cx)}, {int(cy)}) H={height_ratio:.1f}%"
            cv2.putText(vis, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # ======== 기존 토픽 퍼블리시 ========
            # data: [cx, cy, height_ratio, class_id]
            msg_out = Float32MultiArray()
            msg_out.data = [float(cx), float(cy), float(height_ratio), float(cls_id)]
            self.pub_info.publish(msg_out)

            # ======== 가장 큰(높이 비율 최대) 박스 선택 ========
            if main_h_ratio is None or height_ratio > main_h_ratio:
                main_h_ratio = height_ratio
                main_cx = cx

        # ======== 선택된 박스 x, 높이퍼센트 퍼블리시 ========
        if main_cx is not None and main_h_ratio is not None:
            self.pub_x.publish(Float32(data=float(main_cx)))
            self.pub_height.publish(Float32(data=float(main_h_ratio)))
        else:
            # 검출이 없을 때도 값을 내보내 주어 C++ 노드가 즉시 회피 해제 가능
            self.pub_x.publish(Float32(data=-1.0))
            self.pub_height.publish(Float32(data=0.0))

        # ======== 회피 조건 체크 & 경고 시각화 ========
        avoid_triggered = False
        if main_cx is not None and main_h_ratio is not None:
            if (main_h_ratio >= self.avoid_height_thresh and
                    main_cx > self.avoid_x_thresh):
                avoid_triggered = True

        # 기준 x라인(회피 x threshold)을 화면에 그려줌
        if 0 < self.avoid_x_thresh < w:
            cv2.line(vis,
                     (int(self.avoid_x_thresh), 0),
                     (int(self.avoid_x_thresh), h),
                     (0, 0, 255), 2)

        if avoid_triggered:
            warn_text = f"AVOID! x={int(main_cx)}, H={main_h_ratio:.1f}%"
            cv2.putText(vis, warn_text, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                        cv2.LINE_AA)
            # 디버그 로그도 같이
            rospy.loginfo_throttle(1.0,
                                   "[YOLO] AVOID TRIGGER: x=%.1f, H=%.1f%% "
                                   "(thresh: x>%.1f, H>=%.1f%%)",
                                   main_cx, main_h_ratio,
                                   self.avoid_x_thresh,
                                   self.avoid_height_thresh)

        # FPS 계산
        now = time.time()
        fps = 1.0 / (now - self.prev_time) if (now - self.prev_time) > 0 else 0.0
        self.prev_time = now
        cv2.putText(vis, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 이미지 출력
        cv2.imshow("YOLO Detection (bbox + class + center + H% + AVOID)", vis)
        cv2.waitKey(1)


def main():
    node = YoloLaneDetectorNode()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

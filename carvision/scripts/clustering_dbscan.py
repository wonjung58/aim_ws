#!/usr/bin/env python3
import rospy
import math
import numpy as np

from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker
import laser_geometry.laser_geometry as lg


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ClusterFollower:
    def __init__(self):
        rospy.loginfo("ClusterFollower Python Node Started")

        self.sub_scan = rospy.Subscriber("/lidar2D", LaserScan, self.scan_callback)
        self.marker_pub = rospy.Publisher("/dbscan_lines", Marker, queue_size=300)
        self.pub_cloud = rospy.Publisher("/scan_points", PointCloud2, queue_size=300)
        self.pub_speed = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.pub_servo = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)

        self.projector = lg.LaserProjection()

# ============================================================
# DBSCAN (C++ 로직 그대로 복붙)
# ============================================================

        # --- parameters 동일하게 유지 ---
        self.eps = 2.0 ##반경 
        self.min_samples = 2  ##최소 포인트 갯수 ##2
        self.follow_speed = 900 #1500
        self.min_speed = 900
        self.k_yaw = 1.2


    def dbscan(self, points):                 
        n = len(points)
        labels = [-1] * n
        cluster_id = 0

        def dist(a, b):
            return math.hypot(a.x - b.x, a.y - b.y)

        for i in range(n):
            if labels[i] != -1:
                continue

            neighbors = []
            for j in range(n):
                if dist(points[i], points[j]) <= self.eps:
                    neighbors.append(j)

            if len(neighbors) < self.min_samples:
                continue

            cluster_id += 1
            labels[i] = cluster_id

            seed_set = list(neighbors)   

            k = 0
            while k < len(seed_set):
                j = seed_set[k]
                if labels[j] == -1:
                    labels[j] = cluster_id

                    j_neighbors = []
                    for m in range(n):
                        if dist(points[j], points[m]) <= self.eps:
                            j_neighbors.append(m)

                    if len(j_neighbors) >= self.min_samples:
                        seed_set.extend(j_neighbors)       
                k += 1

        return labels

    # ============================================================
    # LaserScan Callback
    # ============================================================
    def scan_callback(self, scan):    ######################################################

        points = []
        angle = scan.angle_min

        # --- 필터링 ---
        for r in scan.ranges:
            if not math.isfinite(r) or r < 0.3 or r > 20.0: #0.13  0.9 ##0.6  20.0
                angle += scan.angle_increment
                continue

            angle_deg = math.degrees(angle)
            if angle_deg < 0:
                angle_deg += 360.0

            # if angle_deg < 330 or angle_deg > 30:  ##60.0   300.0
            #     angle += scan.angle_increment
            #     continue
            angle_deg = (math.degrees(angle) + 360) % 360  # 0~360 보정

            if not (angle_deg >= 270 or angle_deg <= 90):
                angle += scan.angle_increment
                continue


            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append(Point2D(x, y))

            angle += scan.angle_increment

        if not points:
            rospy.logwarn_throttle(1.0, "No valid points — slow forward")  #############
            self.publish_minimal_speed(True)
            return

        # --- PointCloud Publish ---
        cloud = self.projector.projectLaser(scan)
        self.pub_cloud.publish(cloud)

        # --- DBSCAN ---
        labels = self.dbscan(points)

        max_label = max(labels) if labels else 0
        rospy.loginfo_throttle(1.0, "Clusters detected: %d" % max_label)

        left_x_sum = left_y_sum = 0.0
        right_x_sum = right_y_sum = 0.0
        left_cnt = right_cnt = 0

        # --- Cluster 중심 + Marker ---
        for c in range(1, max_label + 1):
            cx = cy = 0.0
            count = 0

            for i, p in enumerate(points):
                if labels[i] == c:
                    cx += p.x
                    cy += p.y
                    count += 1

            if count > 0:
                cx /= count
                cy /= count

                # 좌/우 분리
                if cy > 0:
                    left_cnt += 1
                    left_x_sum += cx
                    left_y_sum += cy
                else:
                    right_cnt += 1
                    right_x_sum += cx
                    right_y_sum += cy

                # Marker Publish
                marker = Marker()
                marker.header.frame_id = scan.header.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "cluster"
                marker.id = c
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = cx
                marker.pose.position.y = cy
                marker.pose.position.z = 0.0
                marker.scale.x = marker.scale.y = marker.scale.z = 0.1
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.lifetime = rospy.Duration(0.1)

                self.marker_pub.publish(marker)

        # --- Target 계산 ---
        if left_cnt > 0 and right_cnt > 0:
            left_x = left_x_sum / left_cnt
            left_y = left_y_sum / left_cnt
            right_x = right_x_sum / right_cnt
            right_y = right_y_sum / right_cnt

            total_cnt = left_cnt + right_cnt
            weight_left = float(right_cnt) / total_cnt
            weight_right = float(left_cnt) / total_cnt

            target_x = weight_left * left_x + weight_right * right_x
            target_y = weight_left * left_y + weight_right * right_y

        elif left_cnt > 0:
            target_x = left_x_sum / left_cnt
            target_y = (left_y_sum / left_cnt) - 0.12

        elif right_cnt > 0:
            target_x = right_x_sum / right_cnt
            target_y = (right_y_sum / right_cnt) + 0.12

        else:
            rospy.logwarn_throttle(1.0, "No clusters → slow forward")
            self.publish_minimal_speed(True)
            return

        # --- Yaw 계산 ---
        yaw = math.atan2(target_y, target_x)
        yaw *= self.k_yaw

        yaw_deg = math.degrees(yaw)

        if yaw_deg < 0:
            steering = 0.545 + (yaw_deg / 25.0) * 0.545
        else:
            steering = 0.545 + (yaw_deg / 20.0) * (1.0 - 0.545)

        steering = max(0.0, min(1.0, steering))

        # --- Speed / Steering Publish ---
        speed_msg = Float64()
        servo_msg = Float64()
        speed_msg.data = self.follow_speed
        servo_msg.data = steering

        self.pub_speed.publish(speed_msg)
        self.pub_servo.publish(servo_msg)

        rospy.loginfo_throttle(1.0, "Speed=%.2f, Steering=%.2f" %
                               (self.follow_speed, steering))

    # ============================================================
    # Minimal Speed
    # ============================================================
    def publish_minimal_speed(self, forward=True):
        speed_msg = Float64()
        servo_msg = Float64()

        if forward:
            speed_msg.data = self.min_speed
            servo_msg.data = 0.545
            rospy.loginfo_throttle(1.0, "→ No target, moving forward slowly")
        else:
            speed_msg.data = 0
            servo_msg.data = 0.545
            rospy.loginfo_throttle(1.0, "→ Stopping")

        self.pub_speed.publish(speed_msg)
        self.pub_servo.publish(servo_msg)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    rospy.init_node("cluster_follower")
    node = ClusterFollower()
    rospy.spin()

#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import deque

# 사전 정보 : 모라이 카메라 위치(X : 2.40, Y : 0.0, Z : 1.18) - 학습 시 카메라 2대 사용, 추론 시 하나만 필요

class LaneDrive:
    def __init__(self):
        rospy.init_node('lane_drive')
        self.bridge = CvBridge()

        # [판제팀에서 수정 필요] 학습된 가중치 모델 경로 설정
        self.model_path = '/home/kante/aim_ws/src/roscpp_morai/src/perception/camera/best.pt'
        self.model = YOLO(self.model_path)

        # Subscriber(모라이 카메라 RGB 토픽), Publisher(판제 필요한 6가지 데이터 담은 배열 형태 토픽)
        self.topic_name = '/camera/rgb'
        self.sub = rospy.Subscriber(self.topic_name, CompressedImage, self.image_callback)
        self.pub = rospy.Publisher('/lane/path', Float32MultiArray, queue_size=1)

        self.LANE_WIDTH = 320  # 한쪽만 보일 때 반대쪽 차선 위치를 추정할 차선 폭(픽셀)
        
        # 스무딩(떨림 보정)을 위한 과거 정보 저장 -> 최근 5프레임 평균 사용
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)

        print(f"차선 판단 주행 시작")

    # 히스토그램을 분석하여 자차와 가장 가까운 차선 픽셀을 추출하는 함수
    def filter_lane_pixels(self, mask, side='left'):
        
        h, w = mask.shape
        mid_x = w // 2
        
        # 좌/우 ROI 영역 설정
        if side == 'left':
            roi = mask[:, :mid_x]
            offset_x = 0
        else:
            roi = mask[:, mid_x:]
            offset_x = mid_x

        # 히스토그램 계산(Y축으로 픽셀을 더해서 X축 분포를 확인)
        bottom_roi = roi[int(h*0.5):, :] # 하늘 말고 가까운 앞쪽 도로만 보고 판단하도록
        histogram = np.sum(bottom_roi, axis=0)
        
        # 인식된 픽셀이 없으면 그냥 빈 배열 반환
        if np.sum(histogram) == 0:
            return np.array([]), np.array([])

        # Peak(가장 픽셀이 많이 뭉친 곳 = 내 차선 중심) 찾기
        peak_x = np.argmax(histogram)
        
        # 마진(Peak 지점 기준 좌우 60 픽셀만 유효하게 - 나름 넓게 잡음)
        margin = 60
        
        # 유효한 픽셀 추출
        nonzero = roi.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 범위 내 픽셀만 남김
        good_inds = ((nonzerox >= (peak_x - margin)) & (nonzerox <= (peak_x + margin)))
        
        # 원본 이미지 좌표계로 변환
        final_x = nonzerox[good_inds] + offset_x
        final_y = nonzeroy[good_inds]

        return final_x, final_y

    # 추출되 픽셀들을 2차 함수로 연결하고, 점선 구간을 잇는 함수(곡선 위해서)
    def fit_lane_curve(self, mask):

        # 필터링된 픽셀들을 가져와서
        lx, ly = self.filter_lane_pixels(mask, side='left')
        rx, ry = self.filter_lane_pixels(mask, side='right')

        left_fit_avg = None
        right_fit_avg = None

        # 왼쪽 차선 피팅 : 2차 함수(x = ay^2 + by + c)
        if len(lx) > 50: # 노이즈 방지 위해 픽셀 50개 이상일 때만
            try:
                current_fit = np.polyfit(ly, lx, 2)
                self.left_fit_history.append(current_fit)
            except: pass

        # 최근 5프레임 평균값 사용(떨림 보정)
        if len(self.left_fit_history) > 0:
            left_fit_avg = np.mean(self.left_fit_history, axis=0)

        # 오른쪽 차선 피팅도 동일하게
        if len(rx) > 50:
            try:
                current_fit = np.polyfit(ry, rx, 2)
                self.right_fit_history.append(current_fit)
            except: pass
        if len(self.right_fit_history) > 0:
            right_fit_avg = np.mean(self.right_fit_history, axis=0)

        return left_fit_avg, right_fit_avg

    def image_callback(self, msg):
        try:
            # 이미지 변환
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = cv_img.shape[:2]
            final_display = cv_img.copy()
            
            # 시선 높이 설정(차량 앞쪽 일정 거리-화면 높이 75% 지점을 목표점을 삼음)
            look_ahead_y = int(h * 0.75) 

            # YOLO 모델 추론(Segmenation)
            results = self.model(cv_img, conf=0.4, verbose=False, retina_masks=True)
            
            if results[0].masks is not None:
                # 마스크 추출 및 통합
                masks = results[0].masks.data.cpu().numpy()
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for m in masks:
                    m_resized = cv2.resize(m, (w, h))
                    combined_mask = np.maximum(combined_mask, m_resized)
                combined_mask = (combined_mask > 0.5).astype(np.uint8) * 255

                # 차선 곡선 계산
                left_fit, right_fit = self.fit_lane_curve(combined_mask)

                # 시각화용 Y좌표 배열
                ploty = np.linspace(look_ahead_y, h-1, num=h-look_ahead_y)

                target_left_x, target_right_x = None, None

                # 왼쪽 차선 그리기(파란색)
                if left_fit is not None:
                    l_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                    target_left_x = int(l_fitx[0])
                    pts_left = np.array([np.transpose(np.vstack([l_fitx, ploty]))])
                    cv2.polylines(final_display, np.int32([pts_left]), False, (255, 0, 0), 3)

                # 오른쪽 차선 그리기(노란색)
                if right_fit is not None:
                    r_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                    target_right_x = int(r_fitx[0])
                    pts_right = np.array([np.transpose(np.vstack([r_fitx, ploty]))])
                    cv2.polylines(final_display, np.int32([pts_right]), False, (0, 255, 255), 3)

                # -----------------------------------------------------------
                # [중심점 계산] (점선 연장 및 추정 로직 포함)
                # -----------------------------------------------------------
                center_x = -1

                # 경우 1 : 양쪽 차선이 모두 검출되면 -> 두 차선의 중점
                if target_left_x is not None and target_right_x is not None:
                    center_x = (target_left_x + target_right_x) / 2
                # 경우 2 : 왼족만 보이면 -> 도로 폭 만큼 띄워서 추정
                elif target_left_x is not None:
                    target_right_x = target_left_x + self.LANE_WIDTH
                    center_x = target_left_x + self.LANE_WIDTH / 2
                # 경우 3 : 오른쪽만 보이면 -> 도로 폭 만큼 띄워서 추정
                elif target_right_x is not None:
                    target_left_x = target_right_x - self.LANE_WIDTH
                    center_x = target_right_x - self.LANE_WIDTH / 2

                # -------------------------------------------------------
                # [데이터 계산]
                # -------------------------------------------------------
                if center_x != -1:
                    center_x = np.clip(center_x, 0, w-1)
                    
                    # 1. Offset - 중앙(0.5) 기준 오차(-0.5 ~ +0.5)
                    norm_center = center_x / w
                    offset = 0.5 - norm_center
                    
                    # 2. Vector & Angle 계산(자차에서 목표점 벡터)
                    car_front_x = w // 2   # 자차 중심(화면 가로 중앙)
                    car_front_y = h        # 자차 앞(화면 맨 아래)
                    
                    target_x = int(center_x)    # 목표점 X
                    target_y = look_ahead_y     # 목표점 Y
                    
                    vector_x = target_x - car_front_x  # + : 우측, - : 좌측
                    vector_y = target_y - car_front_y  # 항상 -(위쪽이라서)
                    
                    angle_rad = math.atan2(vector_x, -vector_y) 
                    angle_deg = math.degrees(angle_rad)         # 조향각 변환

                    # 3. 메시지 발행 (Offset, VecX, VecY, Angle, RedDotX, RedDotY)
                    path_msg = Float32MultiArray()
                    path_msg.data = [
                        float(offset),      # data[0]
                        float(vector_x),    # data[1]
                        float(vector_y),    # data[2]
                        float(angle_deg),   # data[3]
                        float(target_x),    # data[4] (Red Dot X)
                        float(target_y)     # data[5] (Red Dot Y)
                    ]
                    self.pub.publish(path_msg)

                    # -------------------------------------------------------
                    # 시각화
                    # -----------------------------------------------------------
                    # 가장 가까운 차선을 잇는 가로선(분홍색)
                    cv2.line(final_display, (int(target_left_x), look_ahead_y), 
                             (int(target_right_x), look_ahead_y), (255, 0, 255), 3)
                    
                    # 가장 가까운 차선점(노란색)
                    cv2.circle(final_display, (int(target_left_x), look_ahead_y), 8, (0, 255, 255), -1)
                    cv2.circle(final_display, (int(target_right_x), look_ahead_y), 8, (0, 255, 255), -1)

                    # 주행 방향 목표점(빨간색)
                    cv2.circle(final_display, (target_x, target_y), 10, (0, 0, 255), -1)

                    # 조향선 = 녹색선(벡터)
                    cv2.line(final_display, (car_front_x, car_front_y), (target_x, target_y), (0, 255, 0), 5)
                    
                    # 값 확인용
                    cv2.putText(final_display, f"Offset: {offset:.3f}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(final_display, f"Angle : {angle_deg:.1f}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(final_display, f"Target: ({target_x}, {target_y})", (10, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(final_display, "Lost Lane", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(final_display, "No Mask", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Lane Data", final_display)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    try:
        LaneDrive()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

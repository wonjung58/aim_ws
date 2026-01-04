#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage


class StereoPanoramaNode:
    def __init__(self):
        rospy.init_node("stereo_panorama", anonymous=False)
        rospy.loginfo("[stereo_panorama] node started (CompressedImage version)")

        # === 파라미터 ===
        self.left_topic  = rospy.get_param("~left_topic",  "/camera/left/image_raw")
        self.right_topic = rospy.get_param("~right_topic", "/camera/right/image_raw")
        self.init_frames = int(rospy.get_param("~init_frames", 30))  # H 학습에 사용할 프레임 수
        self.mask_ratio  = float(rospy.get_param("~mask_ratio", 0.6))  # 위쪽 몇 %만 사용할지

        rospy.loginfo(f"[stereo_panorama] left_topic  = {self.left_topic}")
        rospy.loginfo(f"[stereo_panorama] right_topic = {self.right_topic}")
        rospy.loginfo(f"[stereo_panorama] init_frames = {self.init_frames}")
        rospy.loginfo(f"[stereo_panorama] mask_ratio  = {self.mask_ratio}")

        # 이미지 버퍼
        self.left_img  = None  # BGR
        self.right_img = None  # BGR

        # ORB 특징점 검출기
        self.detector = cv2.ORB_create(nfeatures=1500)
        rospy.loginfo("[stereo_panorama] Using ORB")

        # Homography 고정 관련 변수
        self.H_candidates = []       # 초기 프레임 동안 모을 H (right->left)
        self.H_pano_fixed = None     # 최종 파노라마용 H (translation 포함)
        self.pano_size    = None     # (w, h)
        self.shift        = None     # [dx, dy] : 파노라마 캔버스에서 왼쪽 이미지 위치

        # 창 생성
        self._init_windows()

        # 구독
        self.sub_left = rospy.Subscriber(
            self.left_topic, CompressedImage, self.left_callback, queue_size=1
        )
        self.sub_right = rospy.Subscriber(
            self.right_topic, CompressedImage, self.right_callback, queue_size=1
        )

    # ---------------------- 초기 창 설정 ----------------------
    def _init_windows(self):
        try:
            cv2.startWindowThread()
        except Exception:
            pass

        cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Left", 640, 360)
        cv2.resizeWindow("Right", 640, 360)
        cv2.resizeWindow("Panorama", 960, 480)

        cv2.moveWindow("Left", 50, 50)
        cv2.moveWindow("Right", 750, 50)
        cv2.moveWindow("Panorama", 400, 500)

    # ---------------------- 콜백들 ----------------------
    def left_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            rospy.logwarn("[stereo_panorama] LEFT imdecode failed")
            return

        self.left_img = img
        rospy.loginfo_throttle(
            1.0,
            f"[stereo_panorama] LEFT shape={img.shape}, mean={np.mean(img):.1f}",
        )

        self.process()

    def right_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            rospy.logwarn("[stereo_panorama] RIGHT imdecode failed")
            return

        self.right_img = img
        rospy.loginfo_throttle(
            1.0,
            f"[stereo_panorama] RIGHT shape={img.shape}, mean={np.mean(img):.1f}",
        )

        self.process()

    # ---------------------- warpWithH (고정 H 사용) ----------------------
    def warp_with_H(self, img_left, img_right, H_pano, pano_size, shift):
        """
        img_left:  BGR (왼쪽)
        img_right: BGR (오른쪽)
        H_pano:    오른쪽 -> 파노라마 캔버스 Homography (translation 포함)
        pano_size: (w, h)
        shift:     [dx, dy] : 파노라마 내에서 왼쪽 이미지의 위치
        """
        w_pano, h_pano = pano_size
        dx, dy = shift

        # 오른쪽 이미지를 큰 캔버스에 워핑
        pano = cv2.warpPerspective(img_right, H_pano, (w_pano, h_pano))

        # 왼쪽 이미지를 해당 위치에 덮어쓰기
        h1, w1 = img_left.shape[:2]
        x0, y0 = dx, dy
        x1, y1 = dx + w1, dy + h1

        # 범위 체크
        if x0 < 0 or y0 < 0 or x1 > w_pano or y1 > h_pano:
            rospy.logwarn_throttle(1.0, "[stereo_panorama] Left image out of bounds in pano")
        else:
            pano[y0:y1, x0:x1] = img_left

        return pano

    # ---------------------- 메인 처리 ----------------------
    def process(self):
        # 둘 다 들어왔을 때만
        if self.left_img is None or self.right_img is None:
            return

        img1 = self.left_img   # LEFT
        img2 = self.right_img  # RIGHT

        # 원본 표시
        try:
            cv2.imshow("Left", img1)
            cv2.imshow("Right", img2)
        except Exception as e:
            rospy.logwarn(f"[stereo_panorama] imshow (Left/Right) error: {e}")

        # 평균 밝기 체크 (완전 깜깜하면 skip)
        mean1 = float(np.mean(img1))
        mean2 = float(np.mean(img2))
        if mean1 < 1.0 or mean2 < 1.0:
            rospy.loginfo_throttle(
                1.0,
                f"[stereo_panorama] Too dark images (mean L={mean1:.1f}, R={mean2:.1f}), skip panorama",
            )
            self._show_panorama(img1)
            return

        # === 1) 이미 고정된 H가 있으면, 그걸로만 파노라마 생성 ===
        if self.H_pano_fixed is not None and self.pano_size is not None and self.shift is not None:
            pano = self.warp_with_H(img1, img2, self.H_pano_fixed, self.pano_size, self.shift)
            self._show_panorama(pano)
            return

        # === 2) 아직 H 고정 전: ORB + 매칭 + Homography 계산 ===
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        h1, w1 = gray1.shape[:2]
        h2, w2 = gray2.shape[:2]

        # 상단 mask (하늘/원거리 위주로 안정적 피처만 사용)
        mask1 = np.zeros_like(gray1, dtype=np.uint8)
        mask2 = np.zeros_like(gray2, dtype=np.uint8)
        top1 = int(self.mask_ratio * h1)
        top2 = int(self.mask_ratio * h2)
        mask1[0:top1, :] = 255
        mask2[0:top2, :] = 255

        try:
            kp1, desc1 = self.detector.detectAndCompute(gray1, mask1)
            kp2, desc2 = self.detector.detectAndCompute(gray2, mask2)
        except Exception as e:
            rospy.logwarn(f"[stereo_panorama] detectAndCompute error: {e}")
            self._show_panorama(img1)
            return

        if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
            rospy.loginfo_throttle(
                1.0,
                f"[stereo_panorama] Not enough keypoints (L={len(kp1)}, R={len(kp2)})",
            )
            self._show_panorama(img1)
            return

        # BF-Hamming + knnMatch
        try:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn = matcher.knnMatch(desc1, desc2, k=2)
        except Exception as e:
            rospy.logwarn(f"[stereo_panorama] matcher error: {e}")
            self._show_panorama(img1)
            return

        # Ratio test
        good_matches = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        rospy.loginfo_throttle(
            1.0, f"[stereo_panorama] good_matches = {len(good_matches)}"
        )

        if len(good_matches) < 4:
            self._show_panorama(img1)
            return

        # Homography (RIGHT -> LEFT)
        src_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]  # RIGHT
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]  # LEFT
        ).reshape(-1, 1, 2)

        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        except Exception as e:
            rospy.logwarn(f"[stereo_panorama] findHomography error: {e}")
            self._show_panorama(img1)
            return

        if H is None:
            rospy.loginfo_throttle(1.0, "[stereo_panorama] Homography is None")
            self._show_panorama(img1)
            return

        # H 후보 저장
        self.H_candidates.append(H)
        rospy.loginfo_throttle(
            1.0,
            f"[stereo_panorama] Collected H candidates: {len(self.H_candidates)}/{self.init_frames}",
        )

        # 현재 프레임에서도 일단 파노라마 만들어서 보여주기
        H_pano_tmp, pano_size_tmp, shift_tmp = self.compute_pano_params(img1, img2, H)
        pano_tmp = self.warp_with_H(img1, img2, H_pano_tmp, pano_size_tmp, shift_tmp)
        self._show_panorama(pano_tmp)

        # 충분히 모였으면 평균 H로 고정
        if len(self.H_candidates) >= self.init_frames:
            H_stack = np.stack(self.H_candidates, axis=0)  # (N, 3, 3)
            H_mean = np.mean(H_stack, axis=0)
            # 마지막 원소로 정규화
            H_mean = H_mean / H_mean[2, 2]

            # 고정 파노라마 파라미터 계산
            H_pano_fixed, pano_size, shift = self.compute_pano_params(img1, img2, H_mean)

            self.H_pano_fixed = H_pano_fixed
            self.pano_size    = pano_size
            self.shift        = shift

            rospy.loginfo("[stereo_panorama] H_pano_fixed locked.")
            rospy.loginfo(f"[stereo_panorama] pano_size = {self.pano_size}, shift = {self.shift}")

    # ---------------------- 파노라마 캔버스/변환 계산 ----------------------
    def compute_pano_params(self, img_left, img_right, H_rl):
        """
        img_left, img_right : BGR
        H_rl : RIGHT -> LEFT Homography (3x3)
        return: H_pano, pano_size(w,h), shift[dx,dy]
        """
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]

        # 왼쪽/오른쪽 코너 좌표
        corners1 = np.float32([[0, 0],
                               [w1, 0],
                               [w1, h1],
                               [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0],
                               [w2, 0],
                               [w2, h2],
                               [0, h2]]).reshape(-1, 1, 2)

        # 오른쪽 코너를 왼쪽 좌표계로 변환
        warped_corners2 = cv2.perspectiveTransform(corners2, H_rl)

        # 두 이미지 코너 모두 합치기
        all_corners = np.concatenate((corners1, warped_corners2), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # 파노라마 캔버스 크기
        pano_w = x_max - x_min
        pano_h = y_max - y_min

        # 왼쪽 이미지를 캔버스 안으로 옮기기 위한 shift
        dx = -x_min
        dy = -y_min
        shift = [dx, dy]

        # translation 포함한 최종 Homography (RIGHT -> pano)
        H_translate = np.array([[1, 0, dx],
                                [0, 1, dy],
                                [0, 0, 1]], dtype=np.float64)
        H_pano = H_translate @ H_rl

        return H_pano, (pano_w, pano_h), shift

    # ---------------------- 파노라마 표시 ----------------------
    def _show_panorama(self, pano_img):
        try:
            cv2.imshow("Panorama", pano_img)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logwarn(f"[stereo_panorama] imshow (Panorama) error: {e}")


def main():
    node = StereoPanoramaNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

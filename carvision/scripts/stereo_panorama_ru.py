#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class StereoPanorama:
    def __init__(self):
        print(">>> 1. before init_node")
        rospy.init_node("stereo_panorama")
        print(">>> 2. after init_node")

        self.bridge = CvBridge()
        self.left_img = None
        self.right_img = None

        print(">>> 3. subscribing topics...")
        rospy.Subscriber("/camera/left/image_raw", CompressedImage, self.left_callback)
        rospy.Subscriber("/camera/right/image_raw", CompressedImage, self.right_callback)

        print(">>> 4. node constructor end")

    def left_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.left_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.try_panorama()

    def right_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.right_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.try_panorama()

    def try_panorama(self):
        print(">>> try_panorama called")

        if self.left_img is None or self.right_img is None:
            print(">>> missing one of the images, skip")
            return

        imgL = self.left_img
        imgR = self.right_img

        print(">>> running Stitcher...")

        # ------- Panorama (Stitcher) -------
        try:
            stitcher = cv2.Stitcher_create()
            status, pano = stitcher.stitch([imgL, imgR])

            if status != cv2.Stitcher_OK:
                print("STITCH ERROR:", status)
                pano = imgL.copy()
            else:
                print(">>> Stitch success")

        except Exception as e:
            print("STITCH EXCEPTION:", e)
            pano = imgL.copy()

        # ------- GUI 출력 -------
        try:
            # Left image
            cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Left", 400, 300)
            cv2.moveWindow("Left", 50, 100)
            cv2.imshow("Left", imgL)

            # Right image
            cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Right", 400, 300)
            cv2.moveWindow("Right", 500, 100)
            cv2.imshow("Right", imgR)

            # Panorama
            cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Panorama", 900, 400)
            cv2.moveWindow("Panorama", 50, 450)
            cv2.imshow("Panorama", pano)

            cv2.waitKey(1)

        except Exception as e:
            print("imshow ERROR:", e)
            return


if __name__ == "__main__":
    print(">>> Run block start")
    StereoPanorama()
    print(">>> After constructor, spinning...")
    rospy.spin()
    print(">>> spin ended")
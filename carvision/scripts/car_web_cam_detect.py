#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 1) YOLO 모델 로드
    model_path = "/root/ws/src/yolo/best.pt"  # 필요하면 수정
    print(f"[YOLO] Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = model.names  # {id: name} 딕셔너리

    # 2) 웹캠 열기 (0번 카메라)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0)")
        return

    # 해상도 설정(원하면)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("YOLO Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Webcam", 960, 540)

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam")
            break

        # 3) YOLO 추론
        # - conf, iou 등은 필요하면 인자로 조절 가능 (예: conf=0.4)
        results = model(frame, verbose=False)

        annotated = frame.copy()

        if len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    # 좌표, 클래스, confidence 꺼내기
                    xyxy = box.xyxy[0].cpu().numpy()   # [x1, y1, x2, y2]
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf   = float(box.conf[0].cpu().numpy())

                    x1, y1, x2, y2 = map(int, xyxy)

                    class_name = class_names.get(cls_id, str(cls_id))
                    label = f"{class_name} {conf:.2f}"

                    # 박스 그리기
                    cv2.rectangle(annotated,(x1, y1),(x2, y2),(0, 255, 0),2)
                    # 라벨 텍스트
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

        # 4) 화면 출력
        cv2.imshow("YOLO Webcam", annotated)

        # q 키 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Bye.")

if __name__ == "__main__":
    main()

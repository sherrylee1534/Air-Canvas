import numpy as np
import cv2
from collections import deque
import time
import os
import HandTrackingModule as htm

folder_path = "Header"
header_list = os.listdir(folder_path)
overlay_list = []
# print(header_list)

for header_path in header_list:
    image = cv2.imread(f'{folder_path}/{header_path}')
    overlay_list.append(image)

# print(len(overlay_list))
header = overlay_list[1]  # Set header as first brush selected

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 720)

detector = htm.HandDetector(detection_confidence=0.85)  # High detection confidence bc we want it to be good in painting

while True:
    # 1.Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image - so if I go right, it shows I go right (more intuitive)

    # 2. Find Hand landmarks
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        # print(landmark_list)
        x8, y8 = landmark_list[8][1:]  # Tip of index finger
        x12, y12 = landmark_list[12][1:]  # Tip of middle finger

        # 3. Check which fingers are up - draw when index finger is up, select when index and middle fingers are up
        fingers = detector.no_fingers_up()
        # print(fingers)


        # 0 - 90
        # 125 - 215
        # 260 - 340
        # 390 - 475
        # 510 - 640

        # 4. If Selection Mode - index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1\
                and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            cv2.rectangle(img, (x8, y8-25), (x12, y12+25), (255, 255, 0), cv2.FILLED)
            print("Selection Mode")

            # Check if we are choosing options at the top of the header
            # if y8 < 62:
            #     if 250 < x8 < 450:
            #         header = overlay_list[0]

        # 5. If Drawing Mode - index finger is up
        elif fingers[1] == 1 \
                and fingers[2] == 0 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            cv2.circle(img, (x8, y8), 15, (255, 255, 0), cv2.FILLED)
            print("Drawing Mode")

        # 6. If Invalid Mode - not in Selection Mode or Drawing Mode
        else:
            print("Invalid Mode")

    # Setting header image
    img[0:62, 0:640] = header
    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

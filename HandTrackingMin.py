import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

# Hands class
mp_hands = mp.solutions.hands
# hands = mpHands.Hands()

# Joints of Hands
mp_draw = mp.solutions.drawing_utils

# Frame rate
prev_time = 0
curr_time = 0

while True:
    success, img = cap.read()  # Gives us our frame rate
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Hands class only uses RGB but OpenCV uses BGR
    results = mp_hands.Hands().process(img_rgb)
    # print(results.multi_hand_landmarks)

    # Extract if we have multiple hands
    if results.multi_hand_landmarks:
        for hands in results.multi_hand_landmarks:
            for i, landmark in enumerate(hands.landmark):
                # print(i, landmark)
                height, width, channel = img.shape
                centre_x, centre_y = int(landmark.x * width), int(landmark.y * height)
                # print(i, centre_x, centre_y)

                if i == 8:
                    cv2.circle(img, (centre_x, centre_y), 15, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hands, mp_hands.HAND_CONNECTIONS)

    # Calculating frame rate
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display frame rate
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

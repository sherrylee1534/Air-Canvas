import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        # Hands class
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detection_confidence,
                                         self.track_confidence)

        # Joints of Hands
        self.mp_draw = mp.solutions.drawing_utils

        # Landmark indexes
        self.tip_idx = [4, 8, 12, 16, 20]  # Only for the tips of each digit

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Hands class only uses RGB but OpenCV uses BGR
        # results = mpHands.Hands().process(imgRGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)

        # Extract if we have multiple hands
        if self.results.multi_hand_landmarks:
            for hands in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hands, self.mp_hands.HAND_CONNECTIONS,
                                                self.mp_draw.DrawingSpec(color=(255, 255, 0)),
                                                self.mp_draw.DrawingSpec(color=(0, 0, 0)))

        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for i, landmark in enumerate(my_hand.landmark):
                # print(i, landmark)
                height, width, channel = img.shape
                centre_x, centre_y = int(landmark.x * width), int(landmark.y * height)
                # print(i, centre_x, centre_y)
                # if i == 8:
                #     print(i, centre_x, centre_y)
                self.landmark_list.append([i, centre_x, centre_y])

                # if draw:
                #     if i == 8:
                #         cv2.circle(img, (centre_x, centre_y), 15, (255, 255, 0), cv2.FILLED)

        return self.landmark_list

    def no_fingers_up(self):
        fingers = []

        # Thumb -- haven't check for left/ right thumb (currently only account for right thumb)
        # Check if tip of thumb is on right or left of other landmarks - tells us if it's opened or closed
        if self.landmark_list[self.tip_idx[0]][1] < self.landmark_list[self.tip_idx[0] - 2][1]:
            fingers.append(1)

        else:
            fingers.append(0)

        # 4 fingers
        # Check if tip of finger is above or below other landmarks - tells us if it's opened or closed
        for i in range(1, 5):
            if self.landmark_list[self.tip_idx[i]][2] < self.landmark_list[self.tip_idx[i] - 2][2]:
                fingers.append(1)

            else:
                fingers.append(0)

        return fingers


def main():
    # Frame rate
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cap.read()  # Gives us our frame rate
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)

        if len(landmark_list) != 0:
            print(landmark_list[8])

        # Calculating frame rate
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display frame rate
        cv2.putText(img, str(int(fps)), (30, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv2.imshow("Image", img)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('a'):
            break


if __name__ == "__main__":
    main()

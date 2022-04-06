import numpy as np
import cv2
import time
import os
import HandTrackingModule as htm


folder_path = "Header"
header_list = os.listdir(folder_path)
overlay_list = []
# print(header_list)

for header_path in header_list:
    header_image = cv2.imread(f'{folder_path}/{header_path}')
    overlay_list.append(header_image)

# Header
header = overlay_list[1]  # Set header as colour 1 selected

########################################################################################################################
# Colours
#                   purple           blue             red            cyan           black
draw_color_list = [(230, 141, 186), (255, 204, 102), (12, 12, 235), (255, 255, 0), (0, 0, 0)]
draw_color = draw_color_list[0]  # Init colour as colour 1

# Brush
brush_thickness = 15
eraser_thickness = 50

# Coordinates
# Header X coordinates - left, right
header_options_x = [(0, 90), (125, 215), (260, 340), (390, 475), (510, 640)]
# Header Y coordinates - top, bottom
header_options_y = [0, 62]
# X and Y coordinates for Drawing Mode
prev_x, prev_y = 0, 0

# Transitions
is_drawing = True  # Set default mode as Drawing
is_selection = False
is_invalid = False
is_selection_to_drawing = False
is_invalid_to_drawing = False
prev_mode_list = ["Drawing", "Selection", "Invalid"]
prev_mode = prev_mode_list[0]  # Set default mode as Drawing

# Timer
curr_time, prev_time = 0, 0

# Reset
is_reset = False
is_reset_once = False
########################################################################################################################

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.HandDetector(detection_confidence=0.95)  # High detection confidence bc we want it to be good in drawing

img_canvas = np.zeros((480, 640, 3), np.uint8)
img_reset = np.zeros((480, 640, 3), np.uint8)

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

        # 4. If Selection Mode - index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1 \
                and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # print("Selection mode")
            is_drawing = False
            is_selection = True
            is_invalid = False

            # Check if we are choosing options at the top of the header
            # option 2 - colour 1 (purple)
            # Put option 2 first - easier to see
            # By default, we start with colour 1, and header also shows colour 1 selected
            if y8 < header_options_y[1]:
                if header_options_x[1][0] < x8 < header_options_x[1][1]:
                    header = overlay_list[1]
                    draw_color = draw_color_list[0]

                # option 3 - colour 2 (blue)
                elif header_options_x[2][0] < x8 < header_options_x[2][1]:
                    header = overlay_list[2]
                    draw_color = draw_color_list[1]

                # option 4 - colour 3 (red)
                elif header_options_x[3][0] < x8 < header_options_x[3][1]:
                    header = overlay_list[3]
                    draw_color = draw_color_list[2]

                # option 5 - eraser (black - will erase everything)
                elif header_options_x[4][0] < x8 < header_options_x[4][1]:
                    header = overlay_list[4]
                    draw_color = draw_color_list[4]

                # option 1 - reset
                elif header_options_x[0][0] < x8 < header_options_x[0][1]:
                    header = overlay_list[0]
                    draw_color = draw_color_list[4]
                    is_reset = True

            if is_reset:
                # Reset image - brute force by drawing a black rectangle over entire screen
                img_canvas = cv2.rectangle(img, (0, 0), (640, 480), draw_color, cv2.FILLED)
                draw_color = draw_color_list[0]  # Reset to default colour
                is_reset = False
                is_reset_once = True

            else:
                cv2.rectangle(img, (x8, y8 - 25), (x12, y12 + 25), draw_color, cv2.FILLED)

            prev_mode = prev_mode_list[1]

        # 5. If Drawing Mode - index finger is up
        elif fingers[1] == 1 \
                and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # print("Drawing mode")
            is_drawing = True
            is_selection = False
            is_invalid = False

            # Fix the jumps in Drawing mode when switching from Selection -> Drawing
            if is_selection is False and prev_mode == "Selection":
                is_selection_to_drawing = True

            # Fix the jumps in Drawing mode when switching from Invalid -> Drawing
            if is_invalid is False and prev_mode == "Invalid":
                is_invalid_to_drawing = True

            cv2.circle(img, (x8, y8), 15, draw_color, cv2.FILLED)

            # Very first frame - instead of drawing a line from (0, 0) to (x8, y8), it will just draw a point at
            # (x8, y8). After that, it will keep drawing as a line
            # Also account for when we switch between the 3 modes
            if prev_x == 0 and prev_y == 0 or is_selection_to_drawing or is_invalid_to_drawing:
                prev_x, prev_y = x8, y8

            # If erasing
            if draw_color == draw_color_list[4]:
                cv2.line(img, (prev_x, prev_y), (x8, y8), draw_color, eraser_thickness)
                cv2.line(img_canvas, (prev_x, prev_y), (x8, y8), draw_color, eraser_thickness)

            else:
                cv2.line(img, (prev_x, prev_y), (x8, y8), draw_color, brush_thickness)
                cv2.line(img_canvas, (prev_x, prev_y), (x8, y8), draw_color, brush_thickness)

            prev_x, prev_y = x8, y8
            prev_mode = prev_mode_list[0]
            is_selection_to_drawing = False
            is_invalid_to_drawing = False

        # 6. If Invalid Mode - not in Selection Mode or Drawing Mode
        else:
            # print("Invalid mode")
            is_drawing = False
            is_selection = False
            is_invalid = True
            prev_mode = prev_mode_list[2]

    # Convert to gray image - black and white
    img_grey = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    # When we inverse it in binary, we essentially create a mask - all the black becomes white and the coloured areas
    # become black
    _, img_inverse = cv2.threshold(img_grey, 50, 255, cv2.THRESH_BINARY_INV)
    # Convert it back to make sure both images have 3 colour channels
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    # All the drawings will show up as black on "Image"
    img = cv2.bitwise_and(img, img_inverse)
    # Now, we just add "Image Canvas" to "Image" to get the colours on the drawings
    img = cv2.bitwise_or(img, img_canvas)

    # Calculating frame rate
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display frame rate
    cv2.putText(img, str(int(fps)), (30, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    # Setting header image
    img[0:62, 0:640] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", img_canvas)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()

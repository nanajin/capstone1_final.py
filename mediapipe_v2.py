import cv2
import mediapipe as mp
import numpy as np
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192)  # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue

    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# pTime = 0
#
# def getAngle(word, x, y, first, mid, last):
#     angle = math.degrees(math.atan2(last[1] - mid[1], last[0] - mid[0]) -
#                          math.atan2(first[1] - mid[1], first[0] - mid[0]))
#     angle = abs(angle)  # 양수로 나오게
#     if angle > 180:
#         angle = (360 - angle)
#     # print(word, angle)
#     if word == "left_arm":
#         cv2.putText(image, 'left_arm '+str(int(angle)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (178, 102, 255), 2)
#     if word == "right_arm":
#         cv2.putText(image, 'right_arm '+str(int(angle)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 128), 2)
#
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#
#   while cap.isOpened():
#     success, image = cap.read()
#
#     cTime = time.time()
#     fps = 1/(cTime-pTime)
#     pTime = cTime
#
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)
#
#     # print(results.pose_landmarks)
#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         left_arm_pose = [0 for i in range(3)]  # [0,0,0]
#         right_arm_pose = [0 for i in range(3)]
#         left_leg_pose = [0 for i in range(3)]
#         right_leg_pose = [0 for i in range(3)]
#         left_knee_pose = [0 for i in range(3)]
#         right_knee_pose = [0 for i in range(3)]
#
#         for id, lm in enumerate(results.pose_landmarks.landmark):
#             h, w, c = image.shape
#
#             word = "left_arm"
#             if id == 11:
#                 left_arm_pose[0] = ((lm.x, lm.y))
#                 # left_shoulder.append((lm.x, lm.y))  # 코드가 너무 길어짐
#                 # Append_list(1, lm.x, lm.y)  # 동시에 실행시키는 법 모름
#             if id == 13:
#                 # left_elbow.append((lm.x, lm.y))
#                 left_arm_pose[1] = ((lm.x, lm.y))
#                 lx, ly = int(lm.x*w), int(lm.y*h)
#                 # Append_list(2, lm.x, lm.y)
#             if id == 15:
#                 # left_wrist.append((lm.x, lm.y))
#                 # Append_list(3, lm.x, lm.y)
#                 left_arm_pose[2] = ((lm.x, lm.y))
#
#             if left_arm_pose[0] !=0 and left_arm_pose[1] !=0 and left_arm_pose[2] !=0:
#                 getAngle(word, lx, ly, left_arm_pose[0], left_arm_pose[1], left_arm_pose[2])
#
#             word = "right_arm"
#             if id == 12:
#                 right_arm_pose[0] = ((lm.x, lm.y))
#                 # left_shoulder.append((lm.x, lm.y))
#                 # Append_list(1, lm.x, lm.y)
#             if id == 14:
#                 # left_elbow.append((lm.x, lm.y))
#                 right_arm_pose[1] = ((lm.x, lm.y))
#                 rx, ry = int(lm.x * w), int(lm.y * h)
#                 # Append_list(2, lm.x, lm.y)
#             if id == 16:
#                 # left_wrist.append((lm.x, lm.y))
#                 # Append_list(3, lm.x, lm.y)
#                 right_arm_pose[2] = ((lm.x, lm.y))
#
#             if right_arm_pose[0] != 0 and right_arm_pose[1] != 0 and right_arm_pose[2] != 0:
#                 getAngle(word, rx, ry, right_arm_pose[0], right_arm_pose[1], right_arm_pose[2])
#
#
#             # cx, cy = int(lm.x*w), int(lm.y*h)
#             # cv2.circle(image, (cx, cy), 10, (255,0,255), cv2.FILLED)
#
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('A3_skeleton', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()


# For video input:  속도 느림...
cap = cv2.VideoCapture("C:/Users/nmj37/Desktop/video.mp4")
pTime = 0

def getAngle(word, x, y, first, mid, last):
    angle = math.degrees(math.atan2(last[1] - mid[1], last[0] - mid[0]) -
                         math.atan2(first[1] - mid[1], first[0] - mid[0]))
    angle = abs(angle)  # 양수로 나오게
    if angle > 180:
        angle = (360 - angle)

    if word == "left_arm":
        cv2.putText(image, 'left_arm '+str(int(angle)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (178, 102, 255), 2)
    if word == "right_arm":
        cv2.putText(image, 'right_arm '+str(int(angle)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 128), 2)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('C:/Users/nmj37/Desktop/output', fourcc, 20, (350, 350))

    while cap.isOpened():
        success, image = cap.read()  # success=>ret  image=>frame

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:

            # out.write(image)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            left_arm_pose = [0 for i in range(3)]  # [0,0,0]
            right_arm_pose = [0 for i in range(3)]
            left_leg_pose = [0 for i in range(3)]
            right_leg_pose = [0 for i in range(3)]
            left_knee_pose = [0 for i in range(3)]
            right_knee_pose = [0 for i in range(3)]

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape

                word = "left_arm"
                if id == 11:
                    left_arm_pose[0] = ((lm.x, lm.y))
                    # left_shoulder.append((lm.x, lm.y))  # 코드가 너무 길어짐
                    # Append_list(1, lm.x, lm.y)  # 동시에 실행시키는 법 모름
                if id == 13:
                    # left_elbow.append((lm.x, lm.y))
                    left_arm_pose[1] = ((lm.x, lm.y))
                    lx, ly = int(lm.x*w), int(lm.y*h)
                    # Append_list(2, lm.x, lm.y)
                if id == 15:
                    # left_wrist.append((lm.x, lm.y))
                    # Append_list(3, lm.x, lm.y)
                    left_arm_pose[2] = ((lm.x, lm.y))

                if left_arm_pose[0] !=0 and left_arm_pose[1] !=0 and left_arm_pose[2] !=0:
                    getAngle(word, lx, ly, left_arm_pose[0], left_arm_pose[1], left_arm_pose[2])

                word = "right_arm"
                if id == 12:
                    right_arm_pose[0] = ((lm.x, lm.y))
                    # left_shoulder.append((lm.x, lm.y))
                    # Append_list(1, lm.x, lm.y)
                if id == 14:
                    # left_elbow.append((lm.x, lm.y))
                    right_arm_pose[1] = ((lm.x, lm.y))
                    rx, ry = int(lm.x * w), int(lm.y * h)
                    # Append_list(2, lm.x, lm.y)
                if id == 16:
                    # left_wrist.append((lm.x, lm.y))
                    # Append_list(3, lm.x, lm.y)
                    right_arm_pose[2] = ((lm.x, lm.y))

                if right_arm_pose[0] != 0 and right_arm_pose[1] != 0 and right_arm_pose[2] != 0:
                    getAngle(word, rx, ry, right_arm_pose[0], right_arm_pose[1], right_arm_pose[2])


                # cx, cy = int(lm.x*w), int(lm.y*h)
                # cv2.circle(image, (cx, cy), 10, (255,0,255), cv2.FILLED)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('A3_skeleton', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
cap.release()
# out.release()
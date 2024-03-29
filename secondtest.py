import cv2
import mediapipe as mp
import time
import argparse

class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS

    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
#                 help="path to our input video")
# ap.add_argument("-o", "--output", required=True,
#                 help="path to our output video")
# ap.add_argument("-s", "--fps", type=int, default=30,
#                 help="set fps of output video")
# ap.add_argument("-b", "--black", type=str, default=False,
#                 help="set black background")
# args = vars(ap.parse_args())

pTime = 0
# black_flag = eval(args["black"])
cap = cv2.VideoCapture("C:/Users/nmj37/Desktop/video.mp4")
# out = cv2.VideoWriter("C:/Users/nmj37/Desktop/output", cv2.VideoWriter_fourcc(*"MJPG"),
#                       fps, (int(cap.get(3)), int(cap.get(4))))

detector = PoseDetector()

while (cap.isOpened()):
    success, img = cap.read()

    if success == False:
        break

    img, p_landmarks, p_connections = detector.findPose(img, False)

    # use black background
    # if black_flag:
    #     img = img * 0

    # draw points
    mp.solutions.drawing_utils.draw_landmarks(img, p_landmarks, p_connections)
    lmList = detector.getPosition(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # out.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
# out.release()
cv2.destroyAllWindows()


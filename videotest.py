import cv2
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.num = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, lmList, draw=True):

        # Get the landmarks
        x1, y1 = lmList[0][1:]
        x2, y2 = lmList[1][1:]
        x3, y3 = lmList[2][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle == 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def main():
    cap = cv2.VideoCapture("C:/Users/nmj37/Desktop/video.mp4")
    cap2 = cv2.VideoCapture("C:/Users/nmj37/Desktop/video.mp4")
    pTime = 0
    detector = poseDetector()
    num = 0
    while True:
        num += 1
        success, img = cap.read()
        success2, img2 = cap2.read()
        img = detector.findPose(img)
        img2 = detector.findPose(img2)
        lmList = detector.findPosition(img, draw=False)
        print(num, lmList)

        if len(lmList) >= 10:
            angle = detector.findAngle(img, lmList)
            print(angle)
            print(len(lmList))


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.imshow("Image2", img2)
        if cv2.waitKey(5) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()



"""
    def getPosition(self, img, draw=True):
        imlist = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w) , int (lm.y * h)
            imlist.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
"""
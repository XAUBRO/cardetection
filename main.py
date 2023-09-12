import cv2
import math
from tracker import *

#create tracker object
#tracker = EuclideanDistTracker()


# capture frames from a video
cap = cv2.VideoCapture('carv.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    

    # Extract region of interest
    roi = frame[340:900,200:1000]

    # Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:

        area= cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
            
            detections.append([x,y,w,h])
   
    print(detections)
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
   
    key = cv2.waitKey(30)
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows
import cv2
import numpy as np

def getContours(img,cThr=[100,100],filter=0,draw=True):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel=np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel=kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel=kernel,iterations=2)
    cv2.imshow('canny',imgThre)
    cv2.waitKey(1)
    contours,hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append(len(approx),area,approx,bbox,i)
            else:
                finalContours.append(len(approx),area,approx,bbox,i)
    finalContours = sorted(finalContours,key= lambda x:x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)
     



webcam = False
path = "1.jpg"
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
img = cv2.imread("D:/program files/ppp.jpg")
getContours(img)
cv2.waitKey(1)
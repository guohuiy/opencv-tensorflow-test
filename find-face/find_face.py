# -*- coding: utf-8 -*-

import cv2
import time

frontalFaceFascadeClf=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

line_color=(255,102,51)
font_color=(255,102,51)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("can not open capture!\n")
    exit()

while True:
    ret, frame=cap.read()

    if not ret:
        print("can not read frame")
        break

    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_img=cv2.equalizeHist(gray_img)
    multipleFaces=frontalFaceFascadeClf.detectMultiScale(gray_img, 1.1, 3, 0, (20, 20))

    
    for (x,y,width,height) in multipleFaces:
        cv2.rectangle(frame,(x,y),(x+width,y+height),line_color,2)
        cv2.putText(frame,"FACE",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,font_color,1,cv2.LINE_AA)
        
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

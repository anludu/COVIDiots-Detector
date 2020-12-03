import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')


if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
  
if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')

if face_cascade.empty():
  raise IOError('Unable to load the face cascade classifier xml file')
  
cap = cv2.VideoCapture(0)
ds_factor = 0.5
flagFace=False
flaNose=False
flaMouth=False

xt=10
yt=10


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        flagFace=True
        xt=x
        yt=y
        break
    
    if flagFace:        
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
        for (xm,ym,wm,hm) in mouth_rects:
            ym = int(ym - 0.15*hm)
            cv2.rectangle(frame, (xm,ym), (xm+wm,ym+hm),(0,255,0), 3)
            flaMouth=True
            break
            
        
        nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
        for (xn,yn,wn,hn) in nose_rects:
            cv2.rectangle(frame, (xn,yn), (xn+wn,yn+hn), (0,255,0), 3)
            flaNose=True
            break
        #If any of the two parts is detected, then it will be a covidiot
        if (flaMouth or flaNose):
            cv2.putText(frame,"COVIDIOT DETECTED", (xt,yt), cv2.FONT_ITALIC, 0.5, (0,0,255))
            
        else:
            cv2.putText(frame,"GOOD CITIZEN", (xt,yt), cv2.FONT_ITALIC, 0.5, (0,255,0))
            
        
        
    flag=False
    flaNose=False
    flaMouth=False

    cv2.imshow('Mouth, Face & Nose Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


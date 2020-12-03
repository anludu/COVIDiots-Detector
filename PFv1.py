import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')


if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
  
if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')
  
cap = cv2.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in faces:
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
            
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (xm,ym,wm,hm) in mouth_rects:
        ym = int(ym - 0.15*hm)
        cv2.rectangle(frame, (xm,ym), (xm+wm,ym+hm), (0,255,0), 3)
        break
    
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (xn,yn,wn,hn) in nose_rects:
        cv2.rectangle(frame, (xn,yn), (xn+wn,yn+hn), (0,255,0), 3)
        break

    cv2.imshow('Mouth & Nose Detector', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


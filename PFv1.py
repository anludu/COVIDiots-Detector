#Libraries used for this project
import numpy as np
import cv2

#Setting pre-trained classifiers for each element of the face desired
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

#Validating that the pre-trained classifiers are ready
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
  
if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')

if face_cascade.empty():
  raise IOError('Unable to load the face cascade classifier xml file')
  
#Staring the streaming of the video 
cap = cv2.VideoCapture(0)
ds_factor = 0.5

#Flags that tells which element of the face is detected
flagFace=False
flaNose=False
flaMouth=False

#initial value of text that shows the result of the scanning position
xt=10
yt=10

#Constantly capturing data
while True:
    
    #Saving the frame captured by the streaming
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    #Frame processing to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Multiscaling image processing for taking measurements from several possible angles
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
     #Searching for the face features 
    for (x,y,w,h) in faces:
        #Labelling the face with a blue rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        flagFace=True
        #The result tag goes with the face frame, labeling the subject on real time
        xt=x
        yt=y
        break
    
    if flagFace:        
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
        for (xm,ym,wm,hm) in mouth_rects:
            ym = int(ym - 0.15*hm)
            #Labelling the mouth with a green rectangle
            cv2.rectangle(frame, (xm,ym), (xm+wm,ym+hm),(0,255,0), 3)
            flaMouth=True
            break
            
        
        nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
        for (xn,yn,wn,hn) in nose_rects:
            #Labelling the nose with a green rectangle
            cv2.rectangle(frame, (xn,yn), (xn+wn,yn+hn), (0,255,0), 3)
            flaNose=True
            break
        #If any of the two parts is detected, then it will be a covidiot
        #Showing the results of the scanning, labeling the subject on real time
        if (flaMouth or flaNose):
            #"Covidiot detected" on red to show the break of the rule/ bad use of the mask
            cv2.putText(frame,"COVIDIOT DETECTED", (xt,yt), cv2.FONT_ITALIC, 0.5, (0,0,255))
            
        else:
            #"Good citizen" on green to show a well use of the mask
            cv2.putText(frame,"GOOD CITIZEN", (xt,yt), cv2.FONT_ITALIC, 0.5, (0,255,0))
            
        
    #Re-initializing the flags for the next scanning    
    flagFace=False
    flaNose=False
    flaMouth=False
    
    #Labelling of the tool
    cv2.imshow('COVIDiots Detector', frame)
    
    #If the key ESC is pressed, the window is closed
    c = cv2.waitKey(1)
    #ESC key decimal's equivalente, in ASCII, is 27
    if c == 27: 
        break

cap.release()
cv2.destroyAllWindows()


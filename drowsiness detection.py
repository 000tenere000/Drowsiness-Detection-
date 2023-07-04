from keras.models import load_model
import numpy as np
from pygame import mixer
import os
import cv2
import time

mixer.init()
sound = mixer.Sound("audio/alarm2.mp3")


face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ["close","open"]

model = load_model('models/drownessDetect.h5')


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
thicc = 3
score = 0
path = os.getcwd()

r_predc = [88]
l_predc = [88]


alive = True
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy6.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while alive:
    success,img = cap.read()

    ih , iw , ic = img.shape

    grayIMG = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    #scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
    #minSize :  Minimum possible object size. Objects smaller than that are ignored.
    faces = face.detectMultiScale(grayIMG,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(grayIMG)
    right_eye = reye.detectMultiScale(grayIMG)

    cv2.rectangle(img,(0,ih-50),(200,ih),(0,255,0),thickness=cv2.FILLED)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) 

    for (x,y,w,h) in right_eye:

        r_eye = img[y:y+h,x:x+w]
        
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        r_predc = np.argmax(model.predict(r_eye),axis=-1)
        
        if r_predc == 1:
            lbl = "open"

        if r_predc == 0 :
            lbl = "close"

        break


    for (x,y,w,h) in left_eye:

        l_eye = img[y:y+h,x:x+w]
       
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        l_predc = np.argmax(model.predict(l_eye),axis=-1)
        
        if l_predc == 1:
            lbl = "open"

        if l_predc == 0 :
            lbl = "close"

        break


    if r_predc == 0 and l_predc == 0:
        score +=1
        cv2.putText(img,"Closed",(10,ih-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
    else:
        score=score-1
        cv2.putText(img,"Open",(10,ih-20), font, 1,(255,255,255),1,cv2.LINE_AA)


    if (score<0):
        score =0
    cv2.putText(img,'Score:'+str(score),(100,ih-20), font, 1,(0,0,255),1,cv2.LINE_AA)

    if score>10:

        try:
            sound.play()
        except:
            pass

        if(thicc<11):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(img,(0,0),(iw,ih),(100,100,255),thicc) 
    out.write(img)
    cv2.imshow('Driver',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
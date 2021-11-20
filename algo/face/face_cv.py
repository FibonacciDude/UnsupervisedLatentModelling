import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

print(face_cascade)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('test.mov')

while 1:
    ret, img = cap.read()
    #exp1, exp2 = 108*3/1080, 192*3/1920
    exp1, exp2 = 300, 400
    img = cv2.resize(img, dsize=(int(exp1), int(exp1)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #for (x,y,w,h) in faces:
    #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #    roi_gray = gray[y:y+h, x:x+w]
    #    roi_color = img[y:y+h, x:x+w]
        
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle((0, 255, 0),(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

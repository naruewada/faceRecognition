from base64 import encode
from importlib.resources import path
from pydoc import classname
from unittest import result
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
className = []
myList = os.listdir(path)
print(myList)
name = 'Unknown'

for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    className.append(os.path.splitext(cl)[0])
print(className)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList

def recordAttendance(name):
    with open('venv/record.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            DateString = now.strftime('%m-%d-%Y,%H:%M:%S')
            f.writelines(f'\n{name},{DateString}')
       



encodeListKnomn = findEncoding(images)
print('Encoding Complete')

capture = cv2.VideoCapture(0)


while True:
    success, img = capture.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeface, faceLocation in zip(encodeCurFrame, facesCurFrame):
        mathes = face_recognition.compare_faces(encodeListKnomn, encodeface)
        faceDistance = face_recognition.face_distance(encodeListKnomn, encodeface)
        matchIndex = np.argmin(faceDistance)
        
        if mathes[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 =  y1*4, x2*4, y2*4, x1*4
        else:
            name = 'UNKNOWN'
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 =  y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img,(x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name , (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0), 2)
        print(name)
        recordAttendance(name)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('x'):      
        break










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


def recordAttendance(name, age, gender):
    with open('venv/record.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        # print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            DateString = now.strftime('%m-%d-%Y,%H:%M:%S')
            f.writelines(f'\n{name},{DateString},{gender},{age}')

def faceBox(faceNet, frame):
    # print(frame)
    frameHight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB = False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    # print("Detection ",detection.shape)
    bboxs = []
    for i in range(detection.shape[2]):
       confidence = detection[0,0,i,2]
    #    print("Confidence ",confidence)
       if confidence > 0.7:
           x1 = int(detection[0,0,i,3]*frameWidth)
           y1 = int(detection[0,0,i,4]*frameHight)
           x2 = int(detection[0,0,i,5]*frameWidth)
           y2 = int(detection[0,0,i,6]*frameHight)
           bboxs.append([x1, y1, x2, y2])
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"


ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

encodeListKnomn = findEncoding(images)
print('Encoding Complete')
video = cv2.VideoCapture(0)
padding = 20
while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for bbox, encodeface, faceLocation in zip(bboxs,encodeCurFrame, facesCurFrame):
        # face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        mathes = face_recognition.compare_faces(encodeListKnomn, encodeface)
        faceDistance = face_recognition.face_distance(encodeListKnomn, encodeface)
        matchIndex = np.argmin(faceDistance)
        
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB = False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        if mathes[matchIndex]:
            name = className[matchIndex].upper()
        else:
            name = 'UNKNOWN'
            print(name)
        label = "{},{},{}".format(name,gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        recordAttendance(name, age, gender)
    cv2.imshow("Face Recognition", frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
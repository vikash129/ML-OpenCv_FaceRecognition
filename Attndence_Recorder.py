import os
import numpy as np
from datetime import datetime,date
import pandas as pd
import cv2
import face_recognition as fr

path = 'Training_images'
imageData = []
imageName = []

for image in os.listdir(path):
    data = cv2.imread(f'{path}/{image}')
    imageData.append(data)
    imageName.append(os.path.splitext(image)[0])#get the name of image without extension



def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        # print(len(encode)) #shape : (180, 320, 3) len : 128
        encodeList.append(encode)
    print("encoding complete")
    return encodeList


TrainedEncodedList = findEncodings(imageData)

detectedPeople = {"Name" :  [] , "Time" : [ ]  , "Date" : [ ] }

def markAttendence(name):
    
    try:
        df = pd.read_csv("attendence_data.csv") ;
    except:
        df = pd.DataFrame(detectedPeople) 
        df.to_csv("attendence_data.csv" , mode='w', index=False);


    nameList  = np.array(df.Name) ;
          
    now = datetime.now()
    dtString = now.strftime("%H:%M:%S")

    today = date.today()
    today = today.strftime("%d/%m/%y")

    if name not in detectedPeople["Name"]:
        detectedPeople["Name"].append(name)
        detectedPeople["Time"].append(dtString)
        detectedPeople["Date"].append(today)

        main2()

        if name not in nameList:
            df2 = pd.DataFrame(detectedPeople) 
            df2.to_csv("attendence_data.csv" , mode='w', index=False);


def drawBox(name , frame , faceLoc):
                faceLoc = np.array(faceLoc) * 4
                y1, x2, y2, x1 = faceLoc    #top-right bottom left

                cv2.rectangle(frame, (x1, y1 ), (x2 , y2),  (0, 255, 0) , 3)
                cv2.rectangle(frame, (x1, y2 - 35  ), (x2  , y2  ), (0, 255,255), cv2.FILLED) #box filled wtih name
                cv2.putText(frame, name, (x1 + 6 , y2 - 6),cv2.FONT_HERSHEY_SIMPLEX, 1 , (255, 0 ,0), 2 )              



def main():
    print("opening webcam")                                     
    cap = cv2.VideoCapture(0)

    while True:
        success,frame = cap.read()

        frame_resize = cv2.resize(frame, (0,0) , None  , (1/4) , (1/4))
        frame_resize = cv2.cvtColor(frame_resize , cv2.COLOR_BGR2RGB)

        face_location_list= fr.face_locations(frame_resize) #[(41, 108, 84, 64)]  y1, x2, y2, x1
        test_encoded_list = fr.face_encodings(frame_resize , face_location_list)

        for test_encoded_face,face_pos in zip(test_encoded_list , face_location_list):
            matches = fr.compare_faces(TrainedEncodedList , test_encoded_face) #[False, False, True] 
            faceDis = fr.face_distance(TrainedEncodedList , test_encoded_face) #[0.64483951 0.60420341 0.5293001 ]

            for matchIndex  in range(len(matches)):
                if matches[matchIndex] and faceDis[matchIndex] < 0.57:
                    name = imageName[matchIndex].capitalize()
                    drawBox(name , frame , face_pos)
                    markAttendence(name)
                   
                
        cv2.imshow('Attendence WebCam' , frame)

        if cv2.waitKey(1) == 27:
            print("saving data complete\nclosing webcam\n")
            cv2.destroyAllWindows()
            break



height = 500 
width = 700 

def main2():
    frame = np.full( ( height , width , 3) , (0,0,0) , dtype= np.uint8)

    cv2.putText(frame , "Attendence Sheet" , (80 , 50) , cv2.FONT_HERSHEY_COMPLEX , 1.6 , (255 , 0 , 0) ,3, cv2.LINE_AA   )   
    
    cv2.putText(frame , "Student Name" , (20 , 110) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 0 , 255) ,1, cv2.LINE_AA   )   
    cv2.putText(frame , "Time" , (330 , 110) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 0 , 255) ,1, cv2.LINE_AA   )   
    cv2.putText(frame , "Presented" , (470 , 110) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 0 , 255) ,1, cv2.LINE_AA   )   
    h = 150
    
    i = 1 
    for name,time in  zip(detectedPeople["Name"]  , detectedPeople["Time"]):
        cv2.putText( frame ,str(i) +" "+ name , (18 , h) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) ,1, cv2.LINE_AA   )   
        cv2.putText(frame , time , (330 , h) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) ,1, cv2.LINE_AA   )   
        i+=1
        h+=30

    cv2.putText(frame , str(len( detectedPeople["Name"])) , (550 , 150) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) ,1, cv2.LINE_AA   )   
    cv2.imshow("Attendence Sheet" , frame )
    

print("process encoding...")
main2()
main()

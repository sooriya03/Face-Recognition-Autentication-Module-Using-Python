#importing all the necessary packages 
from imutils.video import VideoStream
import cv2
import numpy as np
import face_recognition
import os
import speech_recognition as sr
import imutils
import time

#defining a function named as face_authentication

def face_authentication():

#below block is used to retrieve and access dataset photos
    
    global user_name
    global folder_name
    path = "F:\\dataset\\"
    images =[]
    classnames = []
    r = sr.Recognizer()
    for foldername in os.listdir(path):
        folderpath = os.path.join(path, foldername)
        if os.path.isdir(folderpath):
            for filename in os.listdir(folderpath):
                imgpath = os.path.join(folderpath, filename)
                curimg = cv2.imread(imgpath)
                images.append(curimg)
                classnames.append(foldername)

#below code is to encode the dataset

    def findencodings(images):
        encodelist =[]
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodelist.append(encode)
        return encodelist

    encodelistknow = findencodings(images)

#below code is to open web cam and capture a photo to detect a face 
    print("Going to detect face please stand straight to the camera")
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    success, img =cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facescurframe = face_recognition.face_locations(imgs)

#if no face is detected below code is executed 
    if len(facescurframe) == 0:
        print("Face not detected properly please stand and focus onto the camera")
        face_authentication()

    encodecurframe = face_recognition.face_encodings(imgs,facescurframe)

#comparing with dataset and accessing the image name as user name and folder as identification number
    for encodeface,faceloc in zip(encodecurframe,facescurframe):
        matches = face_recognition.compare_faces(encodelistknow,encodeface)
        facedis = face_recognition.face_distance(encodelistknow,encodeface)
        matchindex = np.argmin(facedis)
        name =classnames[matchindex]
        folder_path = 'F:\\dataset\\'+name

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                image_path = os.path.join(folder_path, filename)
                folder_name = os.path.basename(folder_path)
                user_name = os.path.splitext(filename)[0]

# if condition for comparing faces and authenticating users with voice commands     
        if matches[matchindex]:
            print("id_",name,user_name,"Is that You?")
            with sr.Microphone() as source1:
                ch = True
                while ch:
                    print("Spell out yes or no")
                    audio_text = r.listen(source1)
                    try:
                        Decide = r.recognize_google(audio_text)
                        Decision = Decide[::2]
                        ch=False

                    except sr.UnknownValueError:
                        print("Sorry, I did not get that")
            sett = "no"
            if(Decision == sett):
                print("Please Authenticate yourself")
                with sr.Microphone() as source3:
                    ch = True
                    while ch:
                        print("Please Spell out your name")
                        audio_text = r.listen(source3)
                        try:
                            user = r.recognize_google(audio_text)
                            user_name = user[::2]
                            print(user_name)
                            ch=False

                        except sr.UnknownValueError:
                            print("Sorry, I did not get that")
#if it is a new user then it is going to take a picture for further authenticating process
                print("Going to take picture of you please stand straight")           
                parent_folder_path = "F:\\dataset\\"
                last_folder_name = sorted(os.listdir(parent_folder_path), reverse=True)[0]
                sequence_number = int(last_folder_name) + 1
                folder_name = str(sequence_number)
                folder_path = os.path.join(parent_folder_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                face_cascade = cv2.CascadeClassifier('face.xml')

                cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    print("Error opening video capture")
                else:
                    face_detected = False  

                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            print("Error capturing frame")
                            break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                        for (x, y, w, h) in faces:
                            file_name = user_name+ ".jpg"
                            file_path = os.path.join(folder_path, file_name)
                            cv2.imwrite(file_path, frame)
                            face_detected = True  
                            break  

                        cv2.imshow('frame', frame)

                        if cv2.waitKey(1) == ord('q') or face_detected:  
                            break

                    cap.release()
                    cv2.destroyAllWindows()
                    print("Authentication process done")
            else:
                print("OK Thank you", user_name)
#if no matches then again he is a new user so again autheticating process is happening
        elif name not in matches:
            print("Please Authenticate yourself")
            with sr.Microphone() as source:
                ch = True
                while ch:
                    print("Please Spell out your name")
                    audio_text = r.listen(source)
                    try:
                        user1 = r.recognize_google(audio_text)
                        user_name = user1[::2]
                        print(user_name)
                        ch=False

                    except sr.UnknownValueError:
                        print("Sorry, I did not get that")
            print("Going to take picture of you please stand straight")
            parent_folder_path = "F:\\dataset\\"
            last_folder_name = sorted(os.listdir(parent_folder_path), reverse=True)[0]
            sequence_number = int(last_folder_name) + 1
            folder_name = str(sequence_number)
            folder_path = os.path.join(parent_folder_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            face_cascade = cv2.CascadeClassifier('face.xml')

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Error opening video capture")
            else:
                face_detected = False  

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        print("Error capturing frame")
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        file_name = user_name+ ".jpg"
                        file_path = os.path.join(folder_path, file_name)
                        cv2.imwrite(file_path, frame)
                        face_detected = True  
                        break  

                    cv2.imshow('frame', frame)

                    if cv2.waitKey(1) == ord('q') or face_detected:  
                        break

                cap.release()
                cv2.destroyAllWindows()
                print("Authentication process done")


face_authentication()


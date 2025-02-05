# from sklearn.neighbors import KNeighborsClassifier

# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime

# from win32com.client import Dispatch

# # Function to speak a given string (text-to-speech)
# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# # Initialize webcam and face detector
# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('D:/gitt/Face-Recognition-Attendance-System-/data/haarcascade_frontalface_default.xml')

# # Load the previously saved labels (names) and faces data
# with open('data/names.pkl', 'rb') as w:
#     LABELS=pickle.load(w)
# with open('data/faces_data.pkl', 'rb') as f:
#     FACES=pickle.load(f)

# print('Shape of Faces matrix --> ', FACES.shape)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# imgBackground=cv2.imread("D:/gitt/Face-Recognition-Attendance-System-/background.png")

# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
#         output=knn.predict(resized_img)
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#         cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
#         attendance=[str(output[0]), str(timestamp)]
#     imgBackground[162:162 + 480, 55:55 + 640] = frame
#     cv2.imshow("Frame",imgBackground)
#     k=cv2.waitKey(1)
#     if k==ord('o'):
#         speak("Attendance Taken..")
#         time.sleep(5)
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(attendance)
#             csvfile.close()
#         else:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
#             csvfile.close()
#     if k==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()




from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

from win32com.client import Dispatch

# Function to speak a given string (text-to-speech)
def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Initialize webcam and face detector
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('D:/gitt/Face-Recognition-Attendance-System-/data/haarcascade_frontalface_default.xml')

# Load the previously saved labels (names) and faces data
with open('data/names.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Check if the number of labels matches the number of face samples
if len(LABELS) != FACES.shape[0]:
    print(f"Error: The number of labels ({len(LABELS)}) does not match the number of faces ({FACES.shape[0]}).")
    
    # Adjust LABELS to match FACES (either truncate or extend LABELS to match FACES)
    if len(LABELS) < FACES.shape[0]:
        additional_labels = ['Unknown'] * (FACES.shape[0] - len(LABELS))  # Add placeholder labels
        LABELS.extend(additional_labels)
    else:
        LABELS = LABELS[:FACES.shape[0]]  # Truncate extra labels if there are more labels than faces

# Train KNN classifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Background image for UI
imgBackground=cv2.imread("D:/gitt/Face-Recognition-Attendance-System-/background.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)

    # Process each face detected in the frame
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

        # Prepare attendance record       
        attendance=[str(output[0]), str(timestamp)]

    # Overlay the frame onto the background image   
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame",imgBackground)

    # Handle user input
    k=cv2.waitKey(1)

    if k==ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)

        # Write attendance to the file
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            # csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            # csvfile.close()
            
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()


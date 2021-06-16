import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from Spotipy import *
import time

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

music_dist={0:"7LI3zw8HLkjKo5YpvA26KG?si=274ab4cd4cf64401",1:"1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b ",2:"1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b ",3:"1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b ",4:"4kvSlabrnfRCQWfN0MgtgA?si=b36add73b4a74b3a",5:"0deORnapZgrxFY4nsKr9JA?si=7a5aba992ea14c93",6:"1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b "}

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]

def web_cam(): 
    global cap1     
    cap1 = cv2.VideoCapture(0)                                 
    if not cap1.isOpened():                             
        print("Cant open the camera")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, web_cam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       exit()
       
def music_rec():
    # frame2=cv2.imread(music_dist[show_text[0]])
    # pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    # img2=Image.fromarray(frame2)
    # imgtk2=ImageTk.PhotoImage(image=img2)
    # lmain2.imgtk2=imgtk2
    # lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    
    # lmain2.configure(image=imgtk2)
    # lmain2.after(10, music_rec)

    track_ids = getTrackIDs('spotify',music_dist[show_text[0]])
    track_list = []
    for i in range(len(track_ids)):
        time.sleep(.3)
        track_data = getTrackFeatures(track_ids[i])
        track_list.append(track_data)

    df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
    f = Frame(root)
    f.place(x=1000,y=600)
    f.pack(side=RIGHT, padx=75, pady=150)
    pt = Table(f, dataframe=df)
    pt.show()
    
if __name__ == '__main__':
    root=tk.Tk()   

    heading2=Label(root,text="Emotion Music Recommendation",pady=20, font=('arial',45,'bold'),bg='white',fg='#CDCDCD')                                 
    
    heading2.pack()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    
    root.title("Music Recommendation")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    web_cam()
    music_rec()
    root.mainloop()

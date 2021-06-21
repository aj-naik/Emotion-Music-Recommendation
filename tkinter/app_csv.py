import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from pandastable import Table, TableModel
from utils import *
from Spotipy import *

emotion_model = keras.Sequential(
    [
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (48,48,1)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')

    ]
)

emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]

def web_cam(): 
    global cap1     
    cap1 = WebcamVideoStream(src=0).start()                                
    frame1 = cap1.read()
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
    print('---------------- Value ------------', music_dist[show_text[0]])
    df = pd.read_csv(music_dist[show_text[0]])
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

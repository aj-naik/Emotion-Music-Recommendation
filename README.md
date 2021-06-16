# Emotion-Music-Recommendation
Recommending music based on your facial expressions using FER 2013 dataset and Sporify api

# Project Description:
The a emotion recognition model is trained on FER 2013 dataset. It can detect 7 emotions. The project works by getting live video feed from web cam, pass it through the model to get a prediction of emotion. Then according to the emotion predicted, the app will fetch playlist of songs from Spotify through spotipy wrapper and recommend the songs by displaying them on the screen.

# Tech Stack:
- Keras
- Tensorflow
- Spotipy
- Tkinter

# Current condition:
The gui in current state is made with tkinter which leads to a very very slow performance and ocassional freezes. However when it does work it works perfectly as it is intended to.

Tkinter gui is only for prototyping, something made to test the app. Final version of app will NOT be in Tkinter. Rather it will be made with FLASK as served as web app. Flask app development is work in progress

# Project Components:
- Spotipy is a module for establishing connection to and getting tracks from Spotify using Spotipy wrapper
- haarcascade is for face detection
- app_csv is the version of app which displays songs which are already fetched and stored in csv format in 'songs' directory
- app is the version which fetches and displays songs on the go at the moment of prediction without storing it and hence is dynamic

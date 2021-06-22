import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

auth_manager = SpotifyClientCredentials('0fce48595d6147a4a6c6769b331929b3','e2682459a920456f8992fe157ea24112')
sp = spotipy.Spotify(auth_manager=auth_manager)

def getTrackIDs(user, playlist_id):
    track_ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        track_ids.append(track['id'])
    return track_ids

def getTrackFeatures(id):
    track_info = sp.track(id)

    name = track_info['name']
    album = track_info['album']['name']
    artist = track_info['album']['artists'][0]['name']
    # release_date = track_info['album']['release_date']
    # length = track_info['duration_ms']
    # popularity = track_info['popularity']

    track_data = [name, album, artist] #, release_date, length, popularity
    return track_data

# Code for creatinf dataframe of feteched playlist

# track_ids = getTrackIDs('spotify',music_dist[show_text[0]])
# track_list = []
# for i in range(len(track_ids)):
#     time.sleep(.3)
#     track_data = getTrackFeatures(track_ids[i])
#     track_list.append(track_data)

# df = pd.DataFrame(track_list, columns = ['Name','Album','Artist']) # ,'Release_date','Length','Popularity'
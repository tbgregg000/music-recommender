#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("/Users/sammannheimer/Documents/CMU_MSBA/Machine Learning II/Data Project/data.csv")
genre_data = pd.read_csv('/Users/sammannheimer/Documents/CMU_MSBA/Machine Learning II/Data Project/data_by_genres.csv')
year_data = pd.read_csv('/Users/sammannheimer/Documents/CMU_MSBA/Machine Learning II/Data Project/data_by_year.csv')


# In[3]:


print(data.info())


# In[4]:


print(genre_data.info())


# In[5]:


print(year_data.info())


# In[6]:


#pip install -U scikit-learn


# In[18]:


##SCM Added
import scikitplot as skplt
from sklearn.cluster import KMeans # Main analysis package

df = genre_data.select_dtypes(np.number)

clustering_kmeans = KMeans(n_clusters=8, n_init=5, random_state=886)
clustering_kmeans.fit(df)
clustering_kmeans.predict(df)


skplt.cluster.plot_elbow_curve(clustering_kmeans, df, cluster_ranges=range(1, 10))
plt.show()


# In[19]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#was originally 10 clusters
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=3))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)


# In[20]:


genre_data.head()


# In[21]:


# Assuming 'cluster' is the column indicating clusters, and other columns are the data features
cluster_means = genre_data.groupby('cluster').mean()

# Print the average for each feature by cluster
print(cluster_means)


# In[22]:


# Visualizing the Clusters with t-SNE

from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()


# In[29]:


##SCM Added
import scikitplot as skplt
from sklearn.cluster import KMeans # Main analysis package

df = song_embedding

clustering_kmeans = KMeans(n_clusters=8, n_init=5, random_state=886)
clustering_kmeans.fit(df)
clustering_kmeans.predict(df)


skplt.cluster.plot_elbow_curve(clustering_kmeans, df, cluster_ranges=range(1, 10))
plt.show()


# In[30]:


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=6, 
                                   verbose=False))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels


# In[31]:


# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()


# In[13]:


get_ipython().system('pip install spotipy')


# In[14]:


import os

client_id = os.environ.get("4ae38db3abaf490b849dfd04b2a28d64")
client_secret = os.environ.get("1acb2afc77814b2dbd95ce43813dc7c7")

print("Client ID:", client_id)
print("Client Secret:", client_secret)


# In[15]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import os  # Don't forget to import the 'os' module

# Replace with your actual environment variable names
client_id = os.environ["4ae38db3abaf490b849dfd04b2a28d64"]
client_secret = os.environ["1acb2afc77814b2dbd95ce43813dc7c7"]

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# The rest of your code remains the same.

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


# In[ ]:


from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


# In[ ]:


recommend_songs([{'name': 'Come As You Are', 'year':1991},
                {'name': 'Smells Like Teen Spirit', 'year': 1991},
                {'name': 'Lithium', 'year': 1992},
                {'name': 'All Apologies', 'year': 1993},
                {'name': 'Stay Away', 'year': 1993}],  data)


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('./movies.csv')
data = movies_data.head()

combined = ''
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
    combined += movies_data[feature]

vectorizer = TfidfVectorizer()            # to transform the texts to numbers
feature_vectorizer = vectorizer.fit_transform(combined)

similarity = cosine_similarity(feature_vectorizer)     # to find similarities by comaparing every movie to every other movie in the list 

movie_name = input("Enter your favourite movie name: ")

list_of_all_titles = movies_data['title'].to_list()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_movie]))

sorted_similar_movies = sorted(similarity_score, key=lambda x:x[1], reverse=True)

print('Movies suggested for you: \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i <= 30:
        print(i, title_from_index)
        i +=1
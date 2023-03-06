import pandas as pd
import numpy as np


def movie_with_splitted_genre(movie):
    genres = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 
              'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 
              'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir', '(no genres listed)']

    for x in genres:
        movie[x] = 0
    for i in range(len(movie.genres)):
        for x in movie.genres[i].split('|'):
            movie[x][i] = 1

    movie = movie.drop(columns='genres')
    titles = movie.title
    #movie = movie.drop(columns='title')
    return movie, titles


def tag_relevance_movies_creation(tag_name_dataframe, tag_relevance_dataframe):
    movieIds = tag_relevance_dataframe.groupby('movieId')
    tag_columns = []
    for i in range(len(tag_name_dataframe)):
        tag_columns.append(tag_name_dataframe.iloc[i, 0])

    tag_relevance_movies = pd.DataFrame({
        'movieId': movieIds.groups.keys()
    })

    y = len(tag_relevance_movies.columns)
    for i in range(len(tag_columns)):
        tag_relevance_movies.insert(y, tag_columns[i], "")
        y += 1

    for i in range(len(tag_relevance_movies.movieId)):
        x = movieIds.get_group(tag_relevance_movies.iloc[i, 0])
        z = np.array(x.relevance)
        tag_relevance_movies.iloc[i, 1:] = z[:]

    return tag_relevance_movies

def return_new_dataframe():
    path = "csv_files/movies.csv"
    movie_dataframe = pd.read_csv(path)
    movie_splitted_genre = movie_with_splitted_genre(movie_dataframe)[0]

    tag_name_dataframe = pd.read_csv('csv_files/genome-tags.csv')
    tag_relevance_dataframe = pd.read_csv('csv_files/genome-scores.csv')

    tag_relevance_movies = tag_relevance_movies_creation(tag_name_dataframe, tag_relevance_dataframe)

    new_dataframe = pd.merge(movie_splitted_genre, tag_relevance_movies, on='movieId')

    return new_dataframe
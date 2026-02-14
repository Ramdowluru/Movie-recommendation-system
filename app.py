
import streamlit as st
import pickle
import pandas as pd
import requests

# ------------------ CONFIG ------------------ #
API_KEY = "149c8bccd16d8ab6de826b5cfc95d84e"
BASE_URL = "https://api.themoviedb.org/3/movie/"
IMAGE_URL = "https://image.tmdb.org/t/p/w500/"
PLACEHOLDER = "https://via.placeholder.com/500x750?text=No+Image"

st.set_page_config(page_title="Movie Recommender", layout="wide")


# ------------------ LOAD DATA ------------------ #
@st.cache_data
def load_data():
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity


movies, similarity = load_data()


# ------------------ FETCH POSTER ------------------ #
@st.cache_data
def fetch_poster(movie_id):
    try:
        url = f"{BASE_URL}{movie_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        data = response.json()

        if data.get("poster_path"):
            return IMAGE_URL + data["poster_path"]
        else:
            return PLACEHOLDER

    except requests.exceptions.RequestException:
        return PLACEHOLDER


# ------------------ RECOMMEND FUNCTION ------------------ #
def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]   # Top 5 recommendations

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# ------------------ UI ------------------ #
st.title("🎬 Movie Recommendation System")

selected_movie_name = st.selectbox(
    "Select a movie:",
    movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    cols = st.columns(5)

    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"**{names[idx]}**")
            st.image(posters[idx])



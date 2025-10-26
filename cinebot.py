import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
import time

# Custom tokenizer for genres
def genre_tokenizer(genres):
    return genres.split('|')

# Load and preprocess data from MovieLens 20M (sampled for performance)
@st.cache_data
def load_data():
    try:
        # Sample first 100000 rows to optimize performance and memory
        ratings = pd.read_csv('rating.csv', skiprows=1, names=['userId', 'movieId', 'rating', 'timestamp'], nrows=100000)
        movies = pd.read_csv('movie.csv', skiprows=1, names=['movieId', 'title', 'genres'])
        # Ensure unique user-movie pairs and pivot
        ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
        data = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        # Convert columns to strings to avoid type mismatch
        data.columns = data.columns.astype(str)
        return data, movies
    except FileNotFoundError:
        st.error("Dataset files (rating.csv, movie.csv) not found. Please download and place them in the directory.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Build models
@st.cache_data
def build_collaborative_model(data):
    try:
        if data.empty:
            return None
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(data)
        return model
    except Exception as e:
        st.error(f"Error building collaborative model: {str(e)}")
        return None

@st.cache_data
def build_content_model(movies):
    try:
        if movies.empty:
            return None, None, pd.DataFrame()
        # Subsample movies to avoid memory allocation error
        movies = movies.head(10000)
        tfidf = TfidfVectorizer(tokenizer=genre_tokenizer)
        tfidf_matrix = tfidf.fit_transform(movies['genres'])
        content_similarity = cosine_similarity(tfidf_matrix)
        return content_similarity, tfidf, movies
    except Exception as e:
        st.error(f"Error building content model: {str(e)}")
        return None, None, pd.DataFrame()

# Get recommendations
def get_recommendations(user_id, data, collab_model, content_similarity, tfidf, movies, selected_genre, selected_mood, n_recommendations=5):
    try:
        if collab_model is None or data.empty or user_id not in data.index:
            st.warning("Using content-based fallback for recommendations.")
            if not movies.empty:
                genre_mask = movies['genres'].str.contains(selected_genre, na=False)
                mood_bias_map = {'Relaxed': 'Comedy', 'Excited': 'Action', 'Sad': 'Drama', 'Adventurous': 'Sci-Fi'}
                mood_genre = mood_bias_map.get(selected_mood, 'Drama')
                mood_mask = movies['genres'].str.contains(mood_genre, na=False)
                filtered_movies = movies[genre_mask | mood_mask].head(n_recommendations)
                return filtered_movies if not filtered_movies.empty else pd.DataFrame(columns=['movieId', 'title', 'genres'])
            return pd.DataFrame(columns=['movieId', 'title', 'genres'])
        
        distances, indices = collab_model.kneighbors(data.loc[user_id].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
        collab_movie_ids = data.columns[indices.flatten()[1:]]
        
        user_rated_ids = data.loc[user_id][data.loc[user_id] > 0].index
        if len(user_rated_ids) == 0:
            user_rated_ids = movies['movieId'].sample(5).values if not movies.empty else []
        user_indices = movies[movies['movieId'].isin(user_rated_ids)].index
        genre_sim_scores = content_similarity[user_indices].mean(axis=0) if user_indices.size > 0 else np.zeros(len(movies))
        
        genre_bias = 0.2 if movies['genres'].str.contains(selected_genre, na=False).any() else 0
        mood_bias_map = {'Relaxed': 'Comedy', 'Excited': 'Action', 'Sad': 'Drama', 'Adventurous': 'Sci-Fi'}
        mood_genre = mood_bias_map.get(selected_mood, 'Drama')
        mood_mask = movies['genres'].str.contains(mood_genre, na=False)
        genre_sim_scores[mood_mask.values] += 0.3
        genre_sim_scores[movies['genres'].str.contains(selected_genre, na=False).values] += genre_bias
        
        content_indices = np.argsort(-genre_sim_scores)[:n_recommendations]
        content_movie_ids = movies.iloc[content_indices]['movieId'].values
        
        all_ids = list(set(collab_movie_ids).union(set(content_movie_ids)))
        rec_movies = movies[movies['movieId'].isin(all_ids)]
        if rec_movies.empty:
            return pd.DataFrame(columns=['movieId', 'title', 'genres'])
        rec_movies['score'] = rec_movies['movieId'].apply(lambda mid: 
            (genre_sim_scores[movies[movies['movieId'] == mid].index[0]] + 
             (data.loc[user_id, mid] if mid in data.loc[user_id] else 0)) / 2)
        recommendations = rec_movies.nlargest(n_recommendations, 'score')
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

# User management
def load_users():
    try:
        if os.path.exists('users.csv'):
            users = pd.read_csv('users.csv')
            if 'userId' not in users.columns:
                if 'user_id' in users.columns:
                    users = users.rename(columns={'user_id': 'userId'})
                else:
                    users = pd.DataFrame(columns=['userId', 'username', 'password', 'language'])
            return users
        return pd.DataFrame(columns=['userId', 'username', 'password', 'language'])
    except Exception as e:
        st.error(f"Error loading users file: {str(e)}")
        return pd.DataFrame(columns=['userId', 'username', 'password', 'language'])

def save_user(username, password, language):
    try:
        users = load_users()
        new_id = users['userId'].max() + 1 if not users.empty and 'userId' in users.columns else 1
        new_user = pd.DataFrame({'userId': [new_id], 'username': [username], 'password': [password], 'language': [language]})
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv('users.csv', index=False)
        return new_id
    except Exception as e:
        st.error(f"Error saving user: {str(e)}")
        return None

def validate_login(username, password):
    try:
        users = load_users()
        user = users[(users['username'] == username) & (users['password'] == password)]
        return user['userId'].iloc[0] if not user.empty else None
    except Exception as e:
        st.error(f"Error validating login: {str(e)}")
        return None

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.language = None

# Dashing UI theme with navy-purple gradient
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    .main {
        background: linear-gradient(135deg, #1E2A44 0%, #6B4E9E 100%);
        color: #FFFFFF;
        padding: 0;
        min-height: 100vh;
        font-family: 'Roboto', sans-serif;
        position: relative;
        overflow-x: hidden;
    }
    .content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 40px 20px;
        position: relative;
        z-index: 1;
    }
    .title {
        font-size: 3.8em;
        font-weight: 700;
        color: #D4A017;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.4);
    }
    .subtitle {
        font-size: 1.3em;
        color: #E8D4A2;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 500;
    }
    .auth-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    .auth-card {
        background: rgba(30, 42, 68, 0.8);
        padding: 30px;
        border-radius: 15px;
        width: 350px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .input-field {
        width: 100%;
        padding: 12px;
        margin: 10px 0;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #E8D4A2;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    .input-field:focus {
        border-color: #D4A017;
        box-shadow: 0 0 8px rgba(212, 160, 23, 0.5);
        background: rgba(255, 255, 255, 0.1);
    }
    .button {
        background: linear-gradient(135deg, #6B4E9E, #D4A017);
        color: #FFFFFF;
        padding: 12px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
    }
    .button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(107, 78, 158, 0.4);
    }
    .recommendation-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 30px;
    }
    .recommendation-card {
        background: rgba(30, 42, 68, 0.9);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    .sidebar {
        background: rgba(30, 42, 68, 0.7);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #D4A017;
    }
    @media (max-width: 768px) {
        .auth-container { flex-direction: column; align-items: center; }
        .auth-card { width: 90%; }
        .recommendation-grid { grid-template-columns: 1fr; }
        .title { font-size: 2.5em; }
        .subtitle { font-size: 1em; }
    }
    </style>
    """, unsafe_allow_html=True)

# Sign-up and Login UI
if not st.session_state.logged_in:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">CineBot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">Your Movie Companion</h3>', unsafe_allow_html=True)
    time.sleep(1)  # Brief welcome delay
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        with st.form(key='signup_form'):
            st.text_input("Username", key="signup_username")
            st.text_input("Password", type="password", key="signup_password")
            st.selectbox("Language", ["English", "Spanish", "French"], key="signup_language")
            if st.form_submit_button("Sign Up"):
                username = st.session_state.signup_username
                password = st.session_state.signup_password
                language = st.session_state.signup_language
                if username and password:
                    user_id = save_user(username, password, language)
                    if user_id:
                        st.success(f"Account created for {username}! Please log in.")
                    else:
                        st.error("Failed to create account. Try again.")
                else:
                    st.error("Please fill all fields.")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        with st.form(key='login_form'):
            st.text_input("Username", key="login_username")
            st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Log In"):
                username = st.session_state.login_username
                password = st.session_state.login_password
                user_id = validate_login(username, password)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    users = load_users()
                    st.session_state.language = users[users['userId'] == user_id]['language'].iloc[0]
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">CineBot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">Your Movie Companion</h3>', unsafe_allow_html=True)
    st.write(f"Welcome back, {st.session_state.username}! Language: {st.session_state.language}")
    
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        st.header("Your Preferences")
        genre_options = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
        mood_options = ['Relaxed', 'Excited', 'Sad', 'Adventurous']
        selected_genre = st.selectbox("Pick a Genre", genre_options)
        selected_mood = st.selectbox("Select Your Mood", mood_options)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Get Top 5 Recommendations", key="recommend_button"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate loading progress
        for i in range(100):
            time.sleep(0.01)  # Small delay to show progress
            progress_bar.progress(i + 1)
            status_text.text(f"Loading recommendations... {i + 1}%")
        
        data, movies = load_data()
        collab_model = build_collaborative_model(data)
        content_sim, tfidf, movies = build_content_model(movies)
        recommendations = get_recommendations(st.session_state.user_id, data, collab_model, content_sim, tfidf, movies, selected_genre, selected_mood)
        
        progress_bar.empty()
        status_text.empty()
        
        if not recommendations.empty:
            st.markdown('<h3 class="subtitle">Recommended Movies</h3>', unsafe_allow_html=True)
            st.markdown('<div class="recommendation-grid">', unsafe_allow_html=True)
            for _, row in recommendations.iterrows():
                st.markdown(f'<div class="recommendation-card"><strong>{row["title"]}</strong><br><small>{row["genres"]}</small></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No recommendations available. Please try different preferences or ensure data is loaded.")
    
    if st.button("Log Out", key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.language = None
        st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)
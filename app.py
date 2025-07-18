# At the top of your app file
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import requests
from io import BytesIO
from sentiment_pipeline_wrapper import SentimentPipelineWrapper
import matplotlib.pyplot as plt
#import torch

# Load Models
sentiment_wrapper = SentimentPipelineWrapper()

with open('movies_df1.pkl', 'rb') as f:
    movies = pickle.load(f)

with open('similarity1.pkl', 'rb') as f:
    similarity = pickle.load(f)



responsive_style = f"""
<style>
/* Global styles */
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

.main-title {{
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin: 1rem 0;
    color: "#FFD166";
}}

.footer {{
    text-align: center;
    font-size: 0.9rem;
    color: grey;
    margin-top: 2rem;
}}

@media (max-width: 768px) {{
    .main-title {{
        font-size: 1.8rem;
    }}

    .stSelectbox > div {{
        font-size: 0.9rem !important;
    }}

    .stButton > button {{
        font-size: 0.9rem;
        padding: 0.4rem 0.8rem;
    }}
}}

@media (max-width: 480px) {{
    .main-title {{
        font-size: 1.5rem;
    }}

    .stImage > img {{
        max-width: 100%;
        height: auto;
    }}

    .stColumn {{
        width: 100% !important;
        display: block !important;
    }}
}}
</style>
"""
st.markdown(responsive_style, unsafe_allow_html=True)


# Functions remain same...
def recommend_movie(movie_title):
    if movie_title not in movies['Title'].values:
        return []
    index = movies[movies['Title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended = []
    for i in distances[1:6]:
        data = movies.iloc[i[0]]
        recommended.append(data)
    return recommended

def get_movie_reviews(movie_title):
    movie_row = movies[movies['Title'] == movie_title]
    if not movie_row.empty:
        return movie_row['Reviews'].iloc[0]
    return None

def get_poster_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

# App Header
st.markdown('<div class="main-title">Bollywood Movie Recommender & Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown("### Discover similar movies and explore audience sentiments!")

# Movie Search
movie_list = sorted(movies['Title'].unique())
with st.sidebar:

    selected_movie = st.selectbox("Search for a movie", movie_list, index=None, placeholder="Type to search...")


if selected_movie:
    movie_data = movies[movies['Title'] == selected_movie].iloc[0]

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        img = get_poster_image(movie_data['Poster_URL'])
        if img:
            st.image(img, use_container_width=True)
        else:
            st.write("Poster not available")

    with col2:
        st.subheader(movie_data['Title'])
        st.markdown(f"**Year:** `{movie_data.get('Year', 'N/A')}`")
        st.markdown(f"**Rating:** `{movie_data.get('Rating', 'N/A')}`")
        st.markdown(f"**Actors:** {movie_data['Actors']}")
        st.markdown("##### Overview")
        st.write(movie_data['Overview'])

    st.markdown("---")

    # Sentiment Analysis
    st.subheader("Audience Sentiment Analysis")
    reviews_text = get_movie_reviews(selected_movie)
    if reviews_text and reviews_text.strip().lower() != "no reviews available":
        review_list = sentiment_wrapper.split_reviews(reviews_text)
        if review_list:
            sentiments = sentiment_wrapper.analyze_sentiments(review_list)
        else:
            sentiments = []
        #sentiments = sentiment_wrapper.analyze_sentiments(review_list)

        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}

        with st.expander("Click to view individual reviews"):
            for review, label, score in sentiments:
                label_text = label_map.get(label, "Unknown")
                sentiment_counts[label_text] += 1
                st.markdown(f"**Review:** {review}")
                st.markdown(f"**Sentiment:** `{label_text}` (Score: {score:.2f})")
                st.markdown("---")

        # Sentiment Chart
        st.markdown("##### Overall Sentiment Summary")
        fig, ax = plt.subplots(figsize=(2.5,1.5))
        ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['red', 'gray', 'green'])
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    else:
        st.warning("No reviews available.")

    # Recommendations
    st.markdown("---")
    st.subheader("You may also like")
    recommended = recommend_movie(selected_movie)
    for i in range(0, len(recommended), 2):
        col1, col2, col3, col4 = st.columns([1, 2, 1, 2])

        # Movie 1
        if i < len(recommended):
            rec1 = recommended[i]
            with col1:
                img1 = get_poster_image(rec1['Poster_URL'])
                if img1:
                    st.image(img1, use_container_width=True)
                else:
                    st.write("Poster not available")
                
            with col2:
                st.markdown(f"**{rec1['Title']}**")
                st.markdown(f"**Year:** `{rec1.get('Year', 'N/A')}`")
                st.markdown(f"**Rating:** `{rec1.get('Rating', 'N/A')}`")
                st.markdown(f"**Actors:** {rec1['Actors']}")
                with st.expander("🔽 Overview"):
                    st.write(rec1['Overview'])


        # Movie 2 right of movie1
        if i + 1 < len(recommended):
            rec2 = recommended[i + 1]
            with col3:
                img2 = get_poster_image(rec2['Poster_URL'])
                if img2:
                    st.image(img2, use_container_width=True)
                else:
                    st.write("Poster not available")
            with col4:
                st.markdown(f"**{rec2['Title']}**")
                st.markdown(f"**Year:** `{rec2.get('Year', 'N/A')}`")
                st.markdown(f"**Rating:** `{rec2.get('Rating', 'N/A')}`")
                st.markdown(f"**Actors:** {rec2['Actors']}")
                with st.expander("🔽 Overview"):
                    st.write(rec2['Overview'])

st.markdown('<div class="footer">Project by Asma</div>', unsafe_allow_html=True)



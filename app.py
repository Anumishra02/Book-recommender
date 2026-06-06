import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

st.set_page_config(
    page_title="BookSense — AI Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,700;0,800;1,700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stBottom"],
section[data-testid="stMain"] > div,
.main, .main > div {
    background: #0b1020 !important;
    color: #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    visibility: hidden !important;
    display: none !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hero */
.booksense-hero {
    position: relative;
    width: 100%;
    min-height: 420px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 84px 20px 56px;
    overflow: hidden;
    background:
        radial-gradient(circle at top, rgba(79, 70, 229, 0.18), transparent 32%),
        radial-gradient(circle at bottom right, rgba(6, 182, 212, 0.10), transparent 28%),
        linear-gradient(180deg, #0b1020 0%, #090d18 100%);
}

.booksense-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.05) 1px, transparent 1px);
    background-size: 56px 56px;
    mask-image: radial-gradient(circle at center, black 0%, transparent 82%);
    opacity: 0.6;
}

.booksense-hero::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.7), transparent);
}

.particle-field {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 2;
    overflow: hidden;
}

.pt {
    position: absolute;
    opacity: 0;
    animation: pt-float linear infinite;
    filter: saturate(0.7) brightness(0.95);
    font-size: 18px;
}

.pt:nth-child(1){left:5%; top:30%;animation-duration:20s;animation-delay:0s; font-size:22px}
.pt:nth-child(2){left:12%; top:65%;animation-duration:25s;animation-delay:3s; font-size:14px}
.pt:nth-child(3){left:88%; top:20%;animation-duration:22s;animation-delay:1s; font-size:20px}
.pt:nth-child(4){left:93%; top:60%;animation-duration:18s;animation-delay:5s; font-size:12px}
.pt:nth-child(5){left:45%; top:8%;animation-duration:30s;animation-delay:2s; font-size:16px}
.pt:nth-child(6){left:25%; top:80%;animation-duration:24s;animation-delay:4s; font-size:10px}
.pt:nth-child(7){left:75%; top:75%;animation-duration:19s;animation-delay:1.5s; font-size:22px}
.pt:nth-child(8){left:60%; top:15%;animation-duration:26s;animation-delay:6s; font-size:13px}

@keyframes pt-float {
    0% { opacity: 0; transform: translateY(30px) rotate(-5deg); }
    8% { opacity: 0.35; }
    92% { opacity: 0.35; }
    100% { opacity: 0; transform: translateY(-80px) rotate(10deg); }
}

.hero-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .35em;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.75);
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    position: relative;
    z-index: 5;
}

.hero-eyebrow::before,
.hero-eyebrow::after {
    content: '';
    width: 34px;
    height: 1px;
    background: rgba(148, 163, 184, 0.25);
}

.hero-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: clamp(60px, 11vw, 116px) !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    line-height: .93 !important;
    text-align: center !important;
    position: relative;
    z-index: 5;
}

.hero-title .w1 {
    display: block;
    background: linear-gradient(135deg, #e5e7eb 0%, #c7d2fe 45%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-title .w2 {
    display: block;
    font-style: italic;
    margin-top: -6px;
    background: linear-gradient(135deg, #a5f3fc 0%, #60a5fa 50%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 14px;
    font-weight: 400;
    letter-spacing: .04em;
    line-height: 1.7;
    color: rgba(229, 231, 235, 0.55);
    margin-top: 22px;
    text-align: center;
    max-width: 460px;
    position: relative;
    z-index: 5;
}

.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 28px;
    position: relative;
    z-index: 5;
    flex-wrap: wrap;
    justify-content: center;
}

.hero-badge {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #c7d2fe;
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 99px;
    padding: 6px 14px;
    background: rgba(79, 70, 229, 0.08);
}

/* Search area */
.search-wrap {
    max-width: 760px;
    margin: 0 auto;
    padding: 36px 20px 0;
}

.search-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .3em;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.7);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.search-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.25), transparent);
}

[data-testid="stSelectbox"] {
    width: 100% !important;
}

[data-testid="stSelectbox"] label {
    display: none !important;
}

div[data-baseweb="select"] > div {
    background: #111827 !important;
    border: 1px solid #243047 !important;
    border-radius: 14px !important;
    min-height: 52px !important;
    padding: 8px 14px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12) !important;
}

div[data-baseweb="select"] * {
    color: #e5e7eb !important;
    -webkit-text-fill-color: #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}

div[data-baseweb="select"] input {
    color: #e5e7eb !important;
}

div[data-baseweb="popover"],
ul[data-baseweb="menu"],
[role="listbox"] {
    background: #0f172a !important;
    border: 1px solid #243047 !important;
    border-radius: 12px !important;
    box-shadow: 0 16px 48px rgba(0,0,0,0.5) !important;
}

li[role="option"],
div[role="option"] {
    background: transparent !important;
    color: rgba(229, 231, 235, 0.8) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
}

li[role="option"]:hover,
div[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background: rgba(79, 70, 229, 0.14) !important;
    color: #c7d2fe !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: .02em !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    font-size: 13px !important;
    transition: all .22s cubic-bezier(.16,1,.3,1) !important;
}

[data-testid="stButton"]:last-of-type > button {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%) !important;
    color: #ffffff !important;
    padding: 14px 32px !important;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.24) !important;
    width: 100% !important;
}

[data-testid="stButton"]:last-of-type > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 36px rgba(79, 70, 229, 0.34) !important;
}

[data-testid="stButton"]:first-of-type > button {
    background: transparent !important;
    color: #c7d2fe !important;
    border: 1px solid #334155 !important;
    padding: 9px 18px !important;
    font-size: 11px !important;
}

[data-testid="stButton"]:first-of-type > button:hover {
    border-color: rgba(99, 102, 241, 0.55) !important;
    color: #e0e7ff !important;
    background: rgba(79, 70, 229, 0.08) !important;
}

/* Results */
.results-hd {
    text-align: center;
    padding: 48px 20px 20px;
}

.results-hd h2 {
    font-family: 'Playfair Display', serif;
    font-size: 34px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #e5e7eb;
    line-height: 1.2;
}

.results-hd h2 em {
    background: linear-gradient(135deg, #c7d2fe, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-style: italic;
}

.results-hd p {
    font-size: 11px;
    color: rgba(229, 231, 235, 0.35);
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-top: 6px;
}

.orn {
    text-align: center;
    color: rgba(99, 102, 241, 0.35);
    font-size: 18px;
    letter-spacing: 10px;
    margin: 4px 0 16px;
}

[data-testid="column"] {
    padding: 6px !important;
}

.bk-num {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .25em;
    color: rgba(148, 163, 184, 0.65);
    text-align: center;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
}

[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    box-shadow: 0 8px 28px rgba(0,0,0,.38) !important;
    transition: transform .35s cubic-bezier(.16,1,.3,1), box-shadow .35s !important;
    width: 100% !important;
}

[data-testid="stImage"] img:hover {
    transform: translateY(-4px) scale(1.01) !important;
    box-shadow: 0 18px 44px rgba(0,0,0,.48) !important;
}

.book-title-text {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    color: rgba(229, 231, 235, 0.72) !important;
    text-align: center !important;
    line-height: 1.5 !important;
    margin-top: 8px !important;
    word-break: break-word !important;
}

.bs-footer {
    text-align: center;
    padding: 52px 20px 28px;
    border-top: 1px solid rgba(148, 163, 184, 0.08);
    margin-top: 48px;
}

.bs-footer-logo {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(135deg, #e5e7eb, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.bs-footer-meta {
    font-size: 10px;
    color: rgba(229, 231, 235, 0.18);
    letter-spacing: .2em;
    text-transform: uppercase;
    margin-top: 8px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.25); border-radius: 99px; }

[data-testid="stSpinner"] > div {
    border-top-color: #6366f1 !important;
}

[data-testid="stAlert"] {
    background: rgba(79, 70, 229, 0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.20) !important;
    border-radius: 10px !important;
    color: #e0e7ff !important;
}
</style>
""", unsafe_allow_html=True)


class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def _load_pickle(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_poster(self, suggestion):
        try:
            book_pivot = self._load_pickle(self.recommendation_config.book_pivot_serialized_objects)
            final_rating = self._load_pickle(self.recommendation_config.final_rating_serialized_objects)

            poster_url = []
            for book_id in suggestion[0]:
                book_name = book_pivot.index[book_id]
                idx = np.where(final_rating['title'] == book_name)[0][0]
                poster_url.append(final_rating.iloc[idx]['image_url'])

            return poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name):
        try:
            model = self._load_pickle(self.recommendation_config.trained_model_path)
            book_pivot = self._load_pickle(self.recommendation_config.book_pivot_serialized_objects)

            book_id = np.where(book_pivot.index == book_name)[0][0]
            _, suggestion = model.kneighbors(
                book_pivot.iloc[book_id, :].values.reshape(1, -1),
                n_neighbors=6
            )

            books_list = [book_pivot.index[i] for i in suggestion[0]]
            poster_url = self.fetch_poster(suggestion)
            return books_list, poster_url
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.success("✦ Model training completed successfully")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)

            st.markdown("""
            <div class="results-hd">
                <h2>Readers also <em>loved</em></h2>
                <p>Collaborative filtering · 5 personalized picks</p>
            </div>
            <div class="orn">· · · · ·</div>
            """, unsafe_allow_html=True)

            cols = st.columns(5)
            labels = ["01", "02", "03", "04", "05"]

            for i, col in enumerate(cols):
                idx = i + 1
                with col:
                    st.markdown(f'<div class="bk-num">№ {labels[i]}</div>', unsafe_allow_html=True)

                    if idx < len(poster_url):
                        try:
                            st.image(poster_url[idx], use_container_width=True)
                        except Exception:
                            st.markdown("🖼️", unsafe_allow_html=True)

                    if idx < len(recommended_books):
                        st.markdown(
                            f'<div class="book-title-text">{recommended_books[idx]}</div>',
                            unsafe_allow_html=True
                        )

        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    obj = Recommendation()

    st.markdown("""
    <div class="booksense-hero">
        <div class="particle-field">
            <span class="pt">📖</span><span class="pt">✦</span>
            <span class="pt">📚</span><span class="pt">◆</span>
            <span class="pt">🔖</span><span class="pt">✦</span>
            <span class="pt">📕</span><span class="pt">◇</span>
        </div>
        <div class="hero-eyebrow"><span>AI · Powered</span></div>
        <h1 class="hero-title">
            <span class="w1">Book</span>
            <span class="w2">Sense</span>
        </h1>
        <p class="hero-sub">
            Discover your next favourite read through<br>the power of collaborative intelligence
        </p>
        <div class="hero-badges">
            <span class="hero-badge">Collaborative Filtering</span>
            <span class="hero-badge">K-Nearest Neighbours</span>
            <span class="hero-badge">ML Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2.8, 1])
    with mid:
        tc, _ = st.columns([1, 5])
        with tc:
            if st.button("⟳ Train"):
                with st.spinner("Training model..."):
                    obj.train_engine()

        st.markdown("""
        <div class="search-wrap">
            <div class="search-label">Select a book to find similar reads</div>
        </div>
        """, unsafe_allow_html=True)

        book_names = pickle.load(open(os.path.join('templates', 'book_names.pkl'), 'rb'))

        selected_book = st.selectbox(
            "Select a book",
            book_names,
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✦ Find Similar Books", use_container_width=True):
            with st.spinner("Finding your next great read..."):
                obj.recommendations_engine(selected_book)

    st.markdown("""
    <div class="bs-footer">
        <div class="bs-footer-logo">BookSense</div>
        <div class="bs-footer-meta">
            Collaborative Filtering · K-Nearest Neighbours · Built with ♥
        </div>
    </div>
    """, unsafe_allow_html=True)
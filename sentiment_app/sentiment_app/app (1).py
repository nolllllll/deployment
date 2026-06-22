import streamlit as st
import joblib
import re
import os

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🧠",
    layout="centered"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 84px;
    font-weight: bold;
    color: #111827;
    line-height: 1.0;
    margin-bottom:10px;
}

.subtitle {
    text-align: center;
    font-size: 24px;
    color: #6b7280;
    margin-bottom: 50px;
    font-weight: 400;
}

.stTextArea textarea {
    border-radius: 15px;
    font-size: 16px;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    background-color: #4f46e5;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton button:hover {
    background-color: #4338ca;
    color: white;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.positive {
    background-color: #dcfce7;
    color: #166534;
}

.neutral {
    background-color: #fef3c7;
    color: #92400e;
}

.negative {
    background-color: #fee2e2;
    color: #991b1b;
}

.footer {
    text-align: center;
    margin-top: 50px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# BASE DIRECTORY
# =====================================

BASE_DIR = os.path.dirname(__file__)

# =====================================
# LOAD MODEL
# =====================================

model_path = os.path.join(
    BASE_DIR,
    "model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)

# =====================================
# STOPWORDS
# =====================================

factory = StopWordRemoverFactory()

stopwords = set(
    factory.get_stop_words()
)

# =====================================
# LABEL MAPPING
# =====================================

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# =====================================
# PREPROCESSING
# =====================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):

    tokens = text.split()

    filtered_tokens = [
        word for word in tokens
        if word not in stopwords
    ]

    return " ".join(filtered_tokens)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.header("📌 About Project")

    st.write("""
    Aplikasi ini menggunakan:
    
    - Deep Learning (IndoBERT)
    - Streamlit Deployment
    
    Untuk melakukan analisis sentimen review Tokopedia.
    """)

    st.info("Created by Emmanuel Alexander")

# =====================================
# MAIN TITLE
# =====================================

st.markdown(
    '<p class="title">🧠 Sentiment Analysis</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Analisis Sentimen Review Tokopedia menggunakan Deep Learning</p>',
    unsafe_allow_html=True
)

# =====================================
# INPUT
# =====================================

user_input = st.text_area(
    "✍️ Masukkan Review",
    height=180,
    placeholder="Contoh: Produk sangat bagus dan pengiriman cepat..."
)

# =====================================
# BUTTON
# =====================================

if st.button("🔍 Predict Sentiment"):

    if user_input.strip() != "":

        with st.spinner("Menganalisis sentimen..."):

            # preprocessing
            cleaned_text = clean_text(
                user_input
            )

            final_text = remove_stopwords(
                cleaned_text
            )

            # transform
            transformed_text = vectorizer.transform(
                [final_text]
            )

            # prediction
            prediction = model.predict(
                transformed_text
            )[0]

            # label handling
            if prediction in id2label:

                sentiment = id2label[
                    prediction
                ]

            else:

                sentiment = str(
                    prediction
                )

        # =====================================
        # RESULT UI
        # =====================================

        if sentiment == "positive":

            emoji = "😊"

            result_class = "positive"

        elif sentiment == "neutral":

            emoji = "😐"

            result_class = "neutral"

        else:

            emoji = "😡"

            result_class = "negative"

        st.markdown(
            f'''
            <div class="result-box {result_class}">
                {emoji} Hasil Sentiment: {sentiment.upper()}
            </div>
            ''',
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "⚠️ Masukkan teks terlebih dahulu"
        )

# =====================================
# FOOTER
# =====================================

st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit</div>',
    unsafe_allow_html=True
)
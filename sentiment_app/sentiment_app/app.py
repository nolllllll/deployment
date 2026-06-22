import streamlit as st
import joblib
import re
import os

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

# =========================
# BASE DIRECTORY
# =========================

BASE_DIR = os.path.dirname(__file__)

# =========================
# LOAD MODEL & VECTORIZER
# =========================

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

# =========================
# STOPWORDS
# =========================

factory = StopWordRemoverFactory()

stopwords = set(
    factory.get_stop_words()
)

# =========================
# LABEL MAPPING
# =========================

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# =========================
# TEXT PREPROCESSING
# =========================

def clean_text(text):

    # lowercase
    text = text.lower()

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):

    tokens = text.split()

    filtered_tokens = [
        word for word in tokens
        if word not in stopwords
    ]

    return " ".join(filtered_tokens)

# =========================
# STREAMLIT UI
# =========================

st.title("Sentiment Analysis Deployment using Streamlit")
st.write("Analisis sentimen review")

# Input user
user_input = st.text_area(
    "Masukkan Review"
)

# Predict button
if st.button("Predict"):

    if user_input.strip() != "":

        # preprocessing
        cleaned_text = clean_text(
            user_input
        )

        final_text = remove_stopwords(
            cleaned_text
        )

        # TF-IDF transform
        transformed_text = vectorizer.transform(
            [final_text]
        )

        # prediction
        prediction = model.predict(
            transformed_text
        )[0]

        # Handle label mapping
        if prediction in id2label:

            sentiment = id2label[
                prediction
            ]

        else:

            sentiment = str(
                prediction
            )

        # Display result
        st.success(
            f"Hasil Sentiment: {sentiment}"
        )

    else:

        st.warning(
            "Masukkan teks terlebih dahulu"
        )
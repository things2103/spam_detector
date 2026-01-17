# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import re, unicodedata
import nltk
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# =======================
# Konfigurasi halaman
# =======================
st.set_page_config(
    page_title="Deteksi Email Spam",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Deteksi Email Spam")
st.write("Pilih CV, model, masukkan teks email, lalu prediksi spam/ham.")

# =======================
# Download stopwords NLTK
# =======================
nltk.download('stopwords')

# =======================
# Load Kamus Gaul & Stopwords
# =======================
df_kamus = pd.read_csv("kamus_gaul.csv")  # sesuaikan path
kamus_normalisasi = dict(zip(df_kamus['slang'], df_kamus['formal']))
kamus_normalisasi.update({
    "communications": "komunikasi",
    "university": "universitas",
    "mail": "email",
    "file": "berkas"
})

stop_factory = StopWordRemoverFactory()
stopwords_id = set(stop_factory.get_stop_words())
stopwords_en = set(stopwords.words('english'))

additional_stopwords = {
    "gue", "viagra", "per",
    "dpc", "nya", "sih",
    "gas", "pana", "corp", "rice", "london","dear",
    "faks", "ees", "lon", "jul","rxoo","gelling","epa"
}

stopwords_all = stopwords_id.union(stopwords_en).union(additional_stopwords)

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# =======================
# Preprocessing
# =======================
def clean_text_model(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = str(text).lower()
    text = re.sub(r'\b(?:https?://|www\.|http\b|https\b)\S*|\S+@\S+\b', ' ', text)
    text = re.sub(r'\b(from|to|cc|bcc|subject|subjek|re|fw|fwd)\b', ' ', text)
    text = re.sub(r'\b(com|net|org|id|edu|inc|co)\b', ' ', text)
    text = re.sub(r'\b(enron|kaminski|vince|shirley|stinson|houston|hice|adobe|john|david|hou|crenshaw|stanford|ect|ee|eb|etc)\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[_\-‐-―]+', ' ', text)
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith("C"))
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [kamus_normalisasi.get(w, w) for w in words]
    words = [stemmer.stem(w) for w in words]
    words = [w for w in words if w not in stopwords_all and len(w) > 2]
    return " ".join(words)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, kamus_normalisasi, stopwords_all, stemmer):
        self.kamus_normalisasi = kamus_normalisasi
        self.stopwords_all = stopwords_all
        self.stemmer = stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text_model(str(x)) for x in X]

preprocess = TextPreprocessor(kamus_normalisasi, stopwords_all, stemmer)

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    max_features=6000,
    sublinear_tf=True,
    smooth_idf=True,
    norm='l2'
)

# =======================
# Pilih Model
# =======================
MODEL_DIR = "models/"  # folder model di repo

models_cv3 = {
    "NB Baseline": "cv3_nb_baseline.pkl",
    "SVM Baseline": "cv3_svm_baseline.pkl",
    "NB Tuned": "cv3_nb_tuned.pkl",
    "SVM Tuned": "cv3_svm_tuned.pkl",
    "Ensemble Hard Baseline": "cv3_ensemble_hard_baseline.pkl",
    "Ensemble Soft Baseline": "cv3_ensemble_soft_baseline.pkl",
    "Ensemble Hard Tuned": "cv3_ensemble_hard_tuned.pkl",
    "Ensemble Soft Tuned": "cv3_ensemble_soft_tuned.pkl"
}

models_cv5 = {
    "NB Baseline": "cv5_nb_baseline.pkl",
    "SVM Baseline": "cv5_svm_baseline.pkl",
    "NB Tuned": "cv5_nb_tuned.pkl",
    "SVM Tuned": "cv5_svm_tuned.pkl",
    "Ensemble Hard Baseline": "cv5_ensemble_hard_baseline.pkl",
    "Ensemble Soft Baseline": "cv5_ensemble_soft_baseline.pkl",
    "Ensemble Hard Tuned": "cv5_ensemble_hard_tuned.pkl",
    "Ensemble Soft Tuned": "cv5_ensemble_soft_tuned.pkl"
}

cv_choice = st.selectbox("Pilih CV:", ["CV 3", "CV 5"])

if cv_choice == "CV 3":
    model_name = st.selectbox("Pilih model:", list(models_cv3.keys()))
    model_file = models_cv3[model_name]
else:
    model_name = st.selectbox("Pilih model:", list(models_cv5.keys()))
    model_file = models_cv5[model_name]

# =======================
# Input email
# =======================
st.subheader("Masukkan teks email:")
email_input = st.text_area("Ketik email di sini:")

# =======================
# Tombol Prediksi
# =======================
if st.button("📧 Prediksi Spam/Ham"):
    if not email_input.strip():
        st.warning("Masukkan teks email terlebih dahulu!")
        st.stop()

    word_count = len(email_input.strip().split())

    if word_count < 10:
        st.error(
            f"Teks terlalu pendek ({word_count} kata).\n\n"
            "⚠️ Minimal **10 kata** diperlukan agar prediksi valid."
        )
        st.stop()

    # Load model fit
    model_path = MODEL_DIR + model_file
    model = joblib.load(model_path)

    # Prediksi
    texts = [email_input.strip()]
    predictions = model.predict(texts)

    try:
        probabilities = model.predict_proba(texts)
    except:
        probabilities = None

    # Tampilkan hasil
    st.subheader("📊 Hasil Prediksi")
    st.success(f"Model: **{model_name}**")
    st.markdown(f"**Kategori:** {predictions[0].upper()}")

    if probabilities is not None:
        prob_dict = dict(zip(model.classes_, probabilities[0]))
        st.markdown("**Probabilitas:**")
        for k, v in prob_dict.items():
            st.write(f"{k}: {v:.2f}")

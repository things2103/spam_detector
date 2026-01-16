# app.py
import streamlit as st
import joblib
import os

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
# Folder model
# =======================
MODEL_DIR = "models/"

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

# =======================
# Pilih CV & model
# =======================
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

    model_path = os.path.join(MODEL_DIR, model_file)
    
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan: {model_file}")
        st.stop()

    # Load pipeline lengkap (preprocess + TF-IDF + model)
    model = joblib.load(model_path)

    # Prediksi
    predictions = model.predict([email_input.strip()])
    try:
        probabilities = model.predict_proba([email_input.strip()])
    except AttributeError:
        probabilities = None

    # Tampilkan hasil
    st.subheader("📊 Hasil Prediksi")
    st.success(f"Model: **{model_name}**")
    st.markdown(f"**Kategori:** {predictions[0].upper()}")

    if probabilities is not None:
        st.markdown("**Probabilitas:**")
        for k, v in zip(model.classes_, probabilities[0]):
            st.write(f"{k}: {v:.2f}")

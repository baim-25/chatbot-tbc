import streamlit as st
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os
import pickle

# Konfigurasi Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key= API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

#Membaca data dari dataset
data = pd.read_csv("tuberculosis_xray_dataset.csv")

def analisis_gejala(teks_user):
    gejala_tbc_umum = {
        "batuk", "batuk parah", "batuk lebih dari 2 minggu",
        "nyeri dada", "sesak napas", "berkeringat malam", "demam",
        "berat badan turun", "lelah", "darah di dahak"
    }

    teks_user = teks_user.lower()
    cocok = []

    for g in gejala_tbc_umum:
        if g in teks_user:
            cocok.append(g)

    if cocok:
        return (
            f"Gejala yang sesuai dengan TBC ditemukan: {', '.join(cocok)}. "
            "Berdasarkan data, ini merupakan indikator umum dari Tuberkulosis.\n"
            "Namun, untuk memastikan, sebaiknya lakukan pemeriksaan lebih lanjut."
        )
    else:
        return (
            "Gejala yang kamu masukkan belum cukup kuat mengindikasikan TBC "
            "berdasarkan data kami. Mohon masukkan lebih banyak gejala atau lakukan pemeriksaan langsung."
        )

#Load model prediksi dan label encoder
with open("model_tbc.pkl", "rb") as f:
    model_prediksi = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

#fungsi prediksi TBC
def prediksi_tbc(input_user: dict):
    df = pd.DataFrame([input_user])
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    hasil_prediksi = model_prediksi.predict(df)[0]
    proba = model_prediksi.predict_proba(df)[0][hasil_prediksi]
    return hasil_prediksi, proba

# Inisialisasi sesi percakapan
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Judul aplikasi
st.title("🤖 Chatbot Deteksi Dini TBC (Tuberkulosis)")

#Form Gejala
with st.form("form_gejala"):
    st.subheader("📝 Isi Gejala Pasien")

    age = st.number_input("Usia", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    chest_pain = st.selectbox("Nyeri Dada?", ["Yes", "No"])
    cough_severity = st.slider("Tingkat Batuk (1-10)", 1, 10, 5)
    breath = st.slider("Sesak Napas (0-3)", 0, 3, 1)
    fatigue = st.slider("Kelelahan (0-10)", 0, 10, 5)
    weight = st.number_input("Penurunan Berat Badan (kg)", 0.0)
    fever = st.selectbox("Demam", ["Mild", "Moderate", "High"])
    night_sweats = st.selectbox("Keringat Malam", ["Yes", "No"])
    sputum = st.selectbox("Produksi Dahak", ["Low", "Medium", "High"])
    blood = st.selectbox("Darah di Dahak", ["Yes", "No"])
    smoking = st.selectbox("Riwayat Merokok", ["Never", "Current", "Former"])
    history_tb = st.selectbox("Riwayat TBC Sebelumnya", ["Yes", "No"])

    submit = st.form_submit_button("Lakukan Prediksi")

if submit:
    data_input = {
        "Age": age,
        "Gender": gender,
        "Chest_Pain": chest_pain,
        "Cough_Severity": cough_severity,
        "Breathlessness": breath,
        "Fatigue": fatigue,
        "Weight_Loss": weight,
        "Fever": fever,
        "Night_Sweats": night_sweats,
        "Sputum_Production": sputum,
        "Blood_in_Sputum": blood,
        "Smoking_History": smoking,
        "Previous_TB_History": history_tb
    }

    hasil, skor = prediksi_tbc(data_input)
    hasil_str = label_encoders["Class"].inverse_transform([hasil])[0]
    hasil_prediksi_text = f"Prediksi sistem: **{hasil_str}** (Probabilitas: {skor:.2f})"
    st.success(hasil_prediksi_text)

    #Simpan ke chat untuk dimasukan ke promt gemini
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"Gejala saya adalah: {data_input}"
    })
    st.session_state.chat_history.append({
        "role": "system",
        "content": hasil_prediksi_text
    })

# Area percakapan sebelumnya
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input pengguna
user_input = st.chat_input("Tanyakan hal lain seputar TBC...")

if user_input:
    # Simpan pesan user
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Buat prompt fokus ke TBC
    full_prompt = "Kamu adalah chatbot medis yang hanya menjawab tentang Tuberkulosis (TBC). Berikut riwayat percakapan:\n"
    for msg in st.session_state.chat_history:
        role = "Pengguna" if msg["role"] == "user" else "Asisten"
        full_prompt += f"{role}: {msg['content']}\n"
    full_prompt += "Asisten:"

    with st.spinner("Memproses..."):
        response = model.generate_content(full_prompt)
        jawaban = response.text.strip()

    st.chat_message("assistant").markdown(jawaban)
    st.session_state.chat_history.append({"role": "assistant", "content": jawaban})

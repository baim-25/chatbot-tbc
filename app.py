import streamlit as st
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import os
API_KEY = os.getenv("GOOGLE_API_KEY")

# Konfigurasi Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key= API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

print("API Key: ", API_KEY)

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



# Inisialisasi sesi percakapan
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Judul aplikasi
st.title("💬 Chatbot Deteksi Dini TBC (Tuberkulosis)")

# Area percakapan sebelumnya
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input pengguna
user_input = st.chat_input("Masukkan gejala atau jawabanmu...")

if user_input:
    # Simpan pesan user
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Tampilkan analisis berdasarkan dataset sebelum kirim ke Gemini
    hasil_analisis = analisis_gejala(user_input)
    st.info(hasil_analisis)

    # Buat prompt fokus ke TBC
    full_prompt = "Kamu adalah asisten medis digital khusus untuk membantu mengenali gejala Tuberkulosis (TBC).\n"
    full_prompt += "Tugasmu adalah bertanya balik jika gejala masih belum cukup jelas, dan membantu pengguna mengenali kemungkinan TBC serta menyarankan tindakan lanjut.\n\n"
    full_prompt += "Riwayat percakapan:\n"
    # Tambahkan hasil analisis ke prompt
    full_prompt += f"\n\nCatatan tambahan dari analisis gejala dataset:\n{hasil_analisis}\n"

    for msg in st.session_state.chat_history:
        role = "Pengguna" if msg["role"] == "user" else "Asisten"
        full_prompt += f"{role}: {msg['content']}\n"
    full_prompt += "Asisten:"

    with st.spinner("Memproses..."):
        response = model.generate_content(full_prompt)

    asisten_reply = response.text.strip()
    st.chat_message("assistant").markdown(asisten_reply)
    st.session_state.chat_history.append({"role": "assistant", "content": asisten_reply})

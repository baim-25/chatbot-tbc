import streamlit as st
import google.generativeai as genai

# Konfigurasi Gemini API
genai.configure(api_key="AIzaSyB-thn8FtHcwsrvkdFI4jaPJEnI_ywhuUk")

model = genai.GenerativeModel("gemini-2.0-flash")

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

    # Buat prompt fokus ke TBC
    full_prompt = "Kamu adalah asisten medis digital khusus untuk membantu mengenali gejala Tuberkulosis (TBC).\n"
    full_prompt += "Tugasmu adalah bertanya balik jika gejala masih belum cukup jelas, dan membantu pengguna mengenali kemungkinan TBC serta menyarankan tindakan lanjut.\n\n"
    full_prompt += "Riwayat percakapan:\n"
    for msg in st.session_state.chat_history:
        role = "Pengguna" if msg["role"] == "user" else "Asisten"
        full_prompt += f"{role}: {msg['content']}\n"
    full_prompt += "Asisten:"

    with st.spinner("Memproses..."):
        response = model.generate_content(full_prompt)

    asisten_reply = response.text.strip()
    st.chat_message("assistant").markdown(asisten_reply)
    st.session_state.chat_history.append({"role": "assistant", "content": asisten_reply})

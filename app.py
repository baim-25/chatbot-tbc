import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
import pickle

# Konfigurasi Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Load model dan label encoder
with open("model_tbc.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Judul Aplikasi
st.title("🩺 Prediksi Penyakit TBC dan Rekomendasi Pengobatan")
st.markdown("Masukkan gejala berikut untuk mengetahui kemungkinan TBC dan mendapatkan saran medis.")

# Form Input Gejala
with st.form("form_gejala"):
    st.subheader("📝 Isi Gejala Pasien")
    col1, col2 = st.columns(2)
    with col1:
        usia = st.number_input("Usia", min_value=0, max_value=120, step=1)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        nyeri_dada = st.selectbox("Nyeri Dada?", ["No", "Yes"])
        batuk = st.slider("Tingkat Batuk (0-9)", 0, 9, 0)
        sesak_napas = st.slider("Sesak Napas (0-4)", 0, 4, 0)
        kelelahan = st.slider("Kelelahan (0-9)", 0, 9, 0)
    with col2:
        berat_badan = st.number_input("Penurunan Berat Badan (kg)", 0.0)
        demam = st.selectbox("Demam", ["Mild", "Moderate", "High"])
        keringat_malam = st.selectbox("Keringat Malam", ["No", "Yes"])
        dahak = st.selectbox("Produksi Dahak", ["High", "Medium", "Low"])
        dahak_berdarah = st.selectbox("Darah di Dahak", ["No", "Yes"])
        merokok = st.selectbox("Riwayat Merokok", ["Never", "Current", "Former"])
        riwayat_tbc = st.selectbox("Riwayat TBC Sebelumnya", ["Yes", "No"])
    submit = st.form_submit_button("Lakukan Prediksi")

#  Proses Prediksi
if submit:
    data_input = {
        "Age": usia,
        "Gender": jenis_kelamin,
        "Chest_Pain": nyeri_dada,
        "Cough_Severity": batuk,
        "Breathlessness": sesak_napas,
        "Fatigue": kelelahan,
        "Weight_Loss": berat_badan,
        "Fever": demam,
        "Night_Sweats": keringat_malam,
        "Sputum_Production": dahak,
        "Blood_in_Sputum": dahak_berdarah,
        "Smoking_History": merokok,
        "Previous_TB_History": riwayat_tbc
    }

    input_df = pd.DataFrame([data_input])

    # Encoding fitur kategorikal
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Prediksi
    probas = model.predict_proba(input_df)[0]

    # Ambil probabilitas khusus untuk label 'Tuberculosis'
    tbc_index = list(model.classes_).index(label_encoders["Class"].transform(["Tuberculosis"])[0])
    probabilitas_tbc = probas[tbc_index]

    # Ambil label prediksi normal
    hasil_prediksi = model.predict(input_df)[0]
    label_teks = label_encoders["Class"].inverse_transform([hasil_prediksi])[0]

    # Tampilkan Hasil
    st.success(f"🩺 Prediksi: **{label_teks}**")
    st.metric(label="Kemungkinan Tuberculosis", value=f"{probabilitas_tbc:.2%}")


    # Jika hasilnya Normal → minta saran dari Gemini
    if label_teks.lower() == "normal":
        with st.spinner("Meminta saran kesehatan dari Gemini..."):
            prompt_normal = f"""
            Seorang pasien memiliki kemungkinan **Normal** berdasarkan hasil prediksi sistem klasifikasi TBC.
            Pasien mengisi gejala sebagai berikut:

            {data_input}

            Sebagai asisten medsis digital, berikan saran kesehatan untuk menjaga kesehatan pasien berdasarkan gejalanya, tambahkan juga informasi dan pentingnya menjaga kesehatan paru-paru dan mencegah penyakit TBC.
            Gunakan bahasa Indonesia yang sopan, ringkas, dan mudah dipahami masyarakat awam.
            """
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt_normal)
            saran_normal = response.text.strip()
        with st.expander("💡 Lihat Saran Kesehatan dari Gemini"):
            st.markdown(saran_normal)


    # Kirim ke Gemini jika TBC
    if label_teks.lower() == "tuberculosis":
        with st.spinner("Memproses rekomendasi dari Gemini..."):
            prompt = f"""
            Kamu adalah asisten medis digital. Seorang pasien didiagnosis memiliki kemungkinan Tuberkulosis berdasarkan gejala berikut:

            {data_input}

            Berdasarkan informasi tersebut, berikan rekomendasi pengobatan awal yang sesuai untuk penderita Tuberkulosis. Jelaskan secara ringkas namun jelas dalam bahasa Indonesia, sertakan juga catatan bahwa pasien tetap perlu konsultasi ke dokter.
            """
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
            jawaban = response.text.strip()
        with st.expander("💡 Rekomendasi Pengobatan dari Gemini"):
            st.markdown(jawaban)

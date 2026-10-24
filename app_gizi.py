import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =============================================================
# 1. KONFIGURASI HALAMAN
# =============================================================
st.set_page_config(
    page_title="SmartKids - Deteksi Gizi AI",
    page_icon="👶",
    layout="centered"
)

# --- CSS FIX (PERBAIKAN TAMPILAN & DARK MODE) ---
st.markdown("""
    <style>
    /* 1. Background Aplikasi jadi Abu-abu Muda */
    .stApp { background-color: #f0f2f6; }
    
    /* 2. Paksa Semua Teks Penting jadi Warna Gelap (Hitam Abu) agar terbaca */
    h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stText, label { color: #31333F !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #31333F !important; }
    .stCaption { color: #555555 !important; }

    /* 3. Styling Kartu Custom */
    .main-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 4. Tombol */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white !important;
        height: 50px;
        font-size: 18px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover { background-color: #FF2B2B; color: white !important; }
    
    /* 5. Styling untuk pesan peringatan custom */
    .warning-box {
        background-color: #fff3cd; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================
# 2. LOAD MODEL
# =============================================================

@st.cache_resource
def load_models():
    try:
        model = joblib.load("malnutrition_model_final.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None


final_tuned_model, FITUR = load_models()

if final_tuned_model is None:
    st.error(
        "File model tidak ditemukan. Pastikan "
        "'malnutrition_model_final.pkl' dan "
        "'model_features.pkl' tersedia."
    )
    st.stop()

# =============================================================
# 4. TAMPILAN ANTARMUKA (FRONTEND)
# =============================================================

st.title("👶 SmartKids AI Nutrition")
st.markdown("Sistem Cerdas Deteksi Dini Status Gizi Anak (Usia **1–5 Tahun**)")
st.write("") # Spacer

# --- SIDEBAR (INPUT) ---
with st.sidebar:
    st.header("📝 Input Data Anak")
    
    # Input Gender
    gender_option = st.radio("Jenis Kelamin", ("Laki-laki", "Perempuan"))
    gender_code = 'L' if gender_option == "Laki-laki" else 'P'
    
    st.subheader("Usia")
    c1, c2 = st.columns(2)
    with c1: age_year = st.number_input("Tahun", 1, 5, 3) 
    with c2: age_month_extra = st.number_input("Bulan Tambahan", 0, 11, 0)
    
    st.subheader("Pengukuran")
    weight = st.number_input("Berat (kg)", 5.0, 150.0, 15.0, 0.1) 
    height = st.number_input("Tinggi (cm)", 60.0, 200.0, 95.0, 0.1) 
    
    st.markdown("---")
    predict_btn = st.button("🚀 Analisis Status Gizi")

# --- MAIN CONTENT (HASIL) ---
if predict_btn:
    # 1. Validasi Usia
    total_age = (age_year * 12) + age_month_extra
    if total_age < 12 or total_age > 60:
        st.error("❌ ERROR: Aplikasi ini hanya mendukung usia 1 tahun (12 bulan) hingga 5 tahun (60 bulan).")
        st.stop()
    
    
    # 2. Hitung BMI untuk informasi tampilan saja
    BMI = weight / (height / 100) ** 2

    # 3. Siapkan input sesuai urutan fitur model
    gender_num = 0 if gender_code == "L" else 1

    input_data = pd.DataFrame(
        [[total_age, weight, height, gender_num]],
        columns=["Age_months", "Weight_kg", "Height_cm", "Gender"],
    )

    # 4. Prediksi langsung menggunakan pipeline model
    pred_class = final_tuned_model.predict(input_data)[0]

    # 5. Mapping label hasil prediksi
    label_map = {
        "Normal": ("Label dataset: Normal", "#00C040"),
        "Underweight": (
            "Label dataset: Underweight",
            "#FF4B4B",
        ),
        "Overweight": (
            "Label dataset: Overweight",
            "#FFA500",
        ),
    }

    label, color = label_map.get(
        pred_class,
        ("Label tidak dikenal", "#808080"),
    )



    # --- TAMPILAN VISUAL HASIL ---
    st.divider()
    
    # Kotak Judul Hasil
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 12px; text-align: center; border: 2px solid {color}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <h2 style="color: {color} !important; margin:0;">Hasil: {label}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spasi
    
    # Progress Bar BMI
    st.markdown("**Posisi BMI Anak:**")
    bmi_percent = min(max((BMI / 35), 0.05), 1.0)
    st.progress(bmi_percent)
    st.caption(f"BMI Aktual: {BMI:.2f} (Skala referensi visual)")

    
    # Tabs Detail
    tab1, tab2, tab3 = st.tabs([
        "📊 Data & Nutrisi",
        "🍽️ Rencana Makan",
        "💡 Rekomendasi Medis"
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Berat Badan", f"{weight:.1f} kg")
        c2.metric("Tinggi Badan", f"{height:.1f} cm")
        c3.metric("BMI (informasi)", f"{BMI:.2f}")

        st.markdown("### Informasi Model")
        st.write(f"**{label}**")
        st.caption(
            "BMI ditampilkan sebagai informasi perhitungan, "
            "bukan sebagai penentu diagnosis gizi anak."
        )
        st.info(
            "Demo ini memprediksi label pada dataset latihan. "
            "Hasilnya tidak menggantikan penilaian tenaga kesehatan."
        )

    with tab2:
        st.subheader("🍎 Edukasi Pola Makan Seimbang")

        st.markdown("""
        Contoh kelompok makanan untuk edukasi pola makan seimbang:

        - **Karbohidrat:** nasi, kentang, atau sumber karbohidrat lain.
        - **Protein:** telur, ikan, ayam, tahu, atau tempe.
        - **Sayur dan buah:** variasikan jenis dan warnanya.
        - **Cairan:** sediakan air putih sesuai kebutuhan.

        Kebutuhan makan anak berbeda-beda berdasarkan usia,
        pertumbuhan, kondisi kesehatan, dan kebutuhan individual.
        """)

        st.info(
            "Informasi ini bersifat edukatif dan tidak disesuaikan "
            "dengan label prediksi model. Jangan menggunakan label "
            "dataset untuk menentukan diet atau porsi medis anak."
        )

    with tab3:
        st.subheader("Catatan Penggunaan")

        st.warning(
            "Aplikasi ini merupakan proyek edukasi machine learning, "
            "bukan alat diagnosis atau pemberi rekomendasi medis."
        )

        st.markdown(
            "Label prediksi harus ditafsirkan berdasarkan keterbatasan "
            "dataset. Jika ada kekhawatiran tentang pertumbuhan atau "
            "status gizi anak, konsultasikan dengan tenaga kesehatan."
        )

else:
    # Layar Sambutan (Welcome Screen)
    st.markdown("""
    <div style='text-align: center; padding: 50px; background-color: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h3>👋 Selamat Datang!</h3>
        <p>Silakan masukkan data anak di menu sebelah kiri untuk memulai analisis.</p>
        <p style="font-size: 0.8em; color: gray;">Demo edukasi Machine Learning menggunakan Random Forest. Bukan alat diagnosis medis.</p>
    </div>
    """, unsafe_allow_html=True)

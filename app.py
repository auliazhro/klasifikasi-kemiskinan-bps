from pathlib import Path
import os

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from classifier_groq import LABELS, make_client, classify_with_groq

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = ["tahun", "ipm", "tpt_feb", "tpak_feb"]
MODEL_PATH = MODEL_DIR / "model_kemiskinan.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"
EXAMPLES_PATH = MODEL_DIR / "llm_examples.csv"
SAMPLE_PATH = DATA_DIR / "sampel_input_kemiskinan.csv"

load_dotenv()
st.set_page_config(
    page_title="Klasifikasi Kemiskinan Provinsi",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_ml_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_examples():
    return pd.read_csv(EXAMPLES_PATH)

def get_api_key():
    key = os.getenv("GROQ_API_KEY", "")
    try:
        if not key:
            key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    return key

st.title("Klasifikasi Tingkat Kemiskinan Provinsi di Indonesia")
st.caption(
    "Studi kasus menggunakan data BPS 2023–2025 | "
    "Metode Machine Learning dan AI Model Service"
)

if not MODEL_PATH.exists() or not EXAMPLES_PATH.exists():
    st.error(
        "Model belum tersedia. Jalankan `python train_ml.py` terlebih dahulu."
    )
    st.stop()

model = load_ml_model()
examples = load_examples()

with st.sidebar:
    st.header("Tentang aplikasi")
    st.write(
        "Aplikasi ini memperkirakan kategori tingkat kemiskinan "
        "berdasarkan IPM, TPT, TPAK, dan tahun pengamatan."
    )
    st.download_button(
        "⬇️ Download sampel input CSV",
        data=SAMPLE_PATH.read_bytes(),
        file_name="sampel_input_kemiskinan.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.divider()
    st.caption(
        "Kategori Rendah, Sedang, dan Tinggi merupakan label penelitian "
        "yang dibentuk dari persentase penduduk miskin BPS. "
        "Hasil aplikasi bukan klasifikasi resmi BPS."
    )

tab_ml, tab_llm, tab_csv = st.tabs([
    "Prediksi Machine Learning",
    "Prediksi Groq LLM",
    "Prediksi Banyak Data (CSV)",
])

def input_form(prefix):
    with st.form(f"form_{prefix}"):
        col1, col2 = st.columns(2)
        with col1:
            tahun = st.selectbox("Tahun pengamatan", [2023, 2024, 2025], key=f"{prefix}_tahun")
            ipm = st.number_input(
                "Indeks Pembangunan Manusia (IPM)",
                min_value=0.0, max_value=100.0,
                value=75.0, step=0.01, key=f"{prefix}_ipm"
            )
        with col2:
            tpt = st.number_input(
                "Tingkat Pengangguran Terbuka (TPT) Februari (%)",
                min_value=0.0, max_value=100.0,
                value=5.0, step=0.01, key=f"{prefix}_tpt"
            )
            tpak = st.number_input(
                "Tingkat Partisipasi Angkatan Kerja (TPAK) Februari (%)",
                min_value=0.0, max_value=100.0,
                value=68.0, step=0.01, key=f"{prefix}_tpak"
            )
        submitted = st.form_submit_button(
            "Prediksi kategori kemiskinan", use_container_width=True
        )
    return submitted, {
        "tahun": tahun,
        "ipm": ipm,
        "tpt_feb": tpt,
        "tpak_feb": tpak,
    }

with tab_ml:
    st.subheader("Prediksi menggunakan model Machine Learning")
    submitted, values = input_form("ml")
    if submitted:
        input_df = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(input_df)[0]
        st.success(f"Hasil prediksi: **{prediction}**")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write("Probabilitas prediksi:")
            st.bar_chart(pd.DataFrame({
                "Kategori": model.classes_,
                "Probabilitas": probabilities,
            }).set_index("Kategori"))
        st.caption("Model: model library terpilih dari evaluasi train_ml.py.")

with tab_llm:
    st.subheader("Prediksi menggunakan Groq AI Model Service")
    st.write("Masukkan indikator yang sama untuk memperoleh prediksi dari LLM.")
    submitted, values = input_form("groq")
    if submitted:
        api_key = get_api_key()
        if not api_key:
            st.error(
                "GROQ_API_KEY belum ditemukan. Isi file .env untuk lokal "
                "atau Streamlit Secrets saat deployment."
            )
        else:
            try:
                with st.spinner("Groq sedang memproses data..."):
                    client = make_client(api_key)
                    prediction, raw = classify_with_groq(
                        client, examples, values
                    )
                if prediction == "invalid":
                    st.warning(f"Model memberikan format jawaban yang tidak dikenali: {raw}")
                else:
                    st.success(f"Hasil prediksi Groq: **{prediction}**")
            except Exception as exc:
                st.error(f"Prediksi Groq gagal: {exc}")

with tab_csv:
    st.subheader("Prediksi beberapa data menggunakan model Machine Learning")
    st.write(
        "Download sampel CSV dari sidebar. File yang diunggah harus memiliki "
        "kolom tahun, ipm, tpt_feb, dan tpak_feb."
    )
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Kolom wajib belum tersedia: {missing}")
            else:
                batch_input = batch[FEATURES].copy()
                if batch_input.isna().any().any():
                    st.error("CSV memiliki nilai kosong pada fitur input.")
                else:
                    batch["prediksi_kategori"] = model.predict(batch_input)
                    st.dataframe(batch, use_container_width=True)
                    st.download_button(
                        "Download hasil prediksi CSV",
                        data=batch.to_csv(index=False).encode("utf-8-sig"),
                        file_name="hasil_prediksi_kemiskinan.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"File CSV tidak dapat dibaca: {exc}")

st.divider()
st.caption(
    "Catatan metodologi: model dilatih dari data provinsi-tahun BPS. "
    "Prediksi merupakan estimasi pembelajaran mesin, bukan penetapan "
    "status kemiskinan resmi."
)

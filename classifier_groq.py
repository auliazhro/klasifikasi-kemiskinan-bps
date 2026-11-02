
"""
Klasifikasi kategori kemiskinan provinsi di Indonesia
Menggunakan Groq API - GPT-OSS 20B
Modul 2A dan Modul 2B
"""

import os
import re
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ==================================================
# KONFIGURASI
# ==================================================

LABELS = ["Rendah", "Sedang", "Tinggi"]

DEFAULT_MODEL = "openai/gpt-oss-20b"

# Jumlah contoh few-shot untuk setiap kelas
EXAMPLES_PER_CLASS = 3


# ==================================================
# KONEKSI GROQ
# ==================================================

def make_client(api_key=None):

    key = api_key or os.getenv("GROQ_API_KEY")

    if not key:
        raise ValueError(
            "GROQ_API_KEY tidak ditemukan. "
            "Periksa file .env."
        )

    return Groq(
        api_key=key,
        max_retries=0
    )


# ==================================================
# PARSING LABEL
# ==================================================

def parse_label(output):

    if not output:
        return "invalid"

    # Jika output sudah berbentuk dictionary
    if isinstance(output, dict):
        candidate = output.get("label", "")

    else:
        text = str(output).strip()
        candidate = ""

        # Mencoba membaca JSON
        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                candidate = parsed.get("label", "")
            elif isinstance(parsed, str):
                candidate = parsed

        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback jika output bukan JSON
        if not candidate:

            ditemukan = [
                label
                for label in LABELS
                if re.search(
                    rf"\b{re.escape(label)}\b",
                    text,
                    flags=re.IGNORECASE
                )
            ]

            # Hanya menerima satu label yang jelas
            if len(ditemukan) == 1:
                candidate = ditemukan[0]
            else:
                return "invalid"

    candidate = str(candidate).strip().lower()

    for label in LABELS:
        if candidate == label.lower():
            return label

    return "invalid"


# ==================================================
# MEMILIH CONTOH FEW-SHOT
# ==================================================

def select_examples(examples):

    if examples is None or examples.empty:
        raise ValueError(
            "Data few-shot kosong."
        )

    kolom_wajib = [
        "tahun",
        "ipm",
        "tpt_feb",
        "tpak_feb",
        "kategori_kemiskinan"
    ]

    for kolom in kolom_wajib:
        if kolom not in examples.columns:
            raise ValueError(
                f"Kolom {kolom} tidak ditemukan."
            )

    hasil = []

    for label in LABELS:

        kelompok = examples[
            examples["kategori_kemiskinan"]
            .astype(str)
            .str.strip()
            .str.lower()
            == label.lower()
        ]

        if kelompok.empty:
            raise ValueError(
                f"Contoh kelas {label} tidak ditemukan."
            )

        sampel = kelompok.sample(
            n=min(
                EXAMPLES_PER_CLASS,
                len(kelompok)
            ),
            random_state=42
        )

        hasil.append(sampel)

    import pandas as pd

    return pd.concat(
        hasil,
        ignore_index=True
    )


# ==================================================
# FORMAT DATA
# ==================================================

def format_example(row):

    return (
        f"Tahun: {int(row['tahun'])}, "
        f"IPM: {float(row['ipm']):.2f}, "
        f"TPT: {float(row['tpt_feb']):.2f}%, "
        f"TPAK: {float(row['tpak_feb']):.2f}%"
    )


# ==================================================
# KLASIFIKASI GROQ
# ==================================================

def classify_with_groq(
    client,
    examples,
    row,
    model=None
):

    model_id = (
        model
        or os.getenv("GROQ_MODEL")
        or DEFAULT_MODEL
    )

    # Pilih maksimal 3 contoh per kelas
    selected = select_examples(examples)

    examples_text = []

    for i, (_, ex) in enumerate(
        selected.iterrows(),
        start=1
    ):

        examples_text.append(
            f"Contoh {i}: "
            f"{format_example(ex)} "
            f"=> {ex['kategori_kemiskinan']}"
        )

    # Data yang akan diprediksi
    data_input = format_example(row)

    prompt = f"""
Kamu adalah sistem klasifikasi kategori kemiskinan
provinsi di Indonesia.

TUGAS:
Prediksi kategori kemiskinan berdasarkan pola
dari contoh data training.

FITUR INPUT:
1. Tahun
2. IPM
3. Tingkat Pengangguran Terbuka (TPT)
4. Tingkat Partisipasi Angkatan Kerja (TPAK)

CONTOH DATA TRAINING:

{chr(10).join(examples_text)}

DATA YANG AKAN DIPREDIKSI:

{data_input}

KETENTUAN:
- Gunakan hanya empat fitur yang diberikan.
- Pelajari hubungan fitur dengan label dari contoh.
- Jangan menggunakan fitur lain.
- Jangan mengarang data.
- Pilih tepat satu kategori.

LABEL YANG DIPERBOLEHKAN:
Rendah, Sedang, atau Tinggi.

Keluarkan hasil dalam JSON dengan format:
{{"label": "Rendah"}}

Jangan menambahkan penjelasan.
"""

    # Memanggil Groq API
    response = client.chat.completions.create(

        model=model_id,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        reasoning_effort="low",

        reasoning_format="hidden",

        max_completion_tokens=512,

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "klasifikasi_kemiskinan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": LABELS
                        }
                    },
                    "required": ["label"],
                    "additionalProperties": False
                }
            }
        }
    )

    # Memeriksa respons API
    if not response.choices:
        raise RuntimeError(
            "Groq tidak mengembalikan pilihan jawaban."
        )

    choice = response.choices[0]
    message = choice.message

    output = message.content or ""

    # Pemeriksaan jika output kosong
    if not output.strip():

        raise RuntimeError(
            "Output Groq kosong. "
            f"Finish reason: {choice.finish_reason}. "
            f"Refusal: {message.refusal}"
        )

    # Parsing hasil
    label = parse_label(output)

    # Jangan meneruskan prediksi invalid
    if label == "invalid":

        raise ValueError(
            "Format output Groq tidak valid: "
            + repr(output)
        )

    return label, output


import pandas as pd
from pathlib import Path

from classifier_groq import (
    make_client,
    classify_with_groq
)

print("=" * 45)
print("PENGUJIAN SATU PREDIKSI GROQ")
print("=" * 45)

# Membaca data training
examples_path = Path("models/llm_examples.csv")

if not examples_path.exists():
    examples_path = Path("outputs/train_data.csv")

# Membaca data testing
test_path = Path("outputs/test_data.csv")

if not examples_path.exists():
    raise FileNotFoundError("Data training tidak ditemukan.")

if not test_path.exists():
    raise FileNotFoundError("Data testing tidak ditemukan.")

examples = pd.read_csv(examples_path)
test = pd.read_csv(test_path)

# Memastikan data tersedia
if len(test) == 0:
    raise ValueError("Data testing kosong.")

# Mengambil satu data testing
row = test.iloc[0]

print("\nData yang diuji:")
print(row[
    ["tahun", "ipm", "tpt_feb", "tpak_feb"]
])

# Koneksi Groq
client = make_client()

print("\nMenghubungi Groq API...")

# Prediksi
prediksi, output = classify_with_groq(
    client,
    examples,
    row
)

print("\nHASIL PREDIKSI:", prediksi)
print("OUTPUT ASLI:", output)

if "kategori_kemiskinan" in row.index:
    print(
        "LABEL SEBENARNYA:",
        row["kategori_kemiskinan"]
    )

print("\nPENGUJIAN SELESAI")

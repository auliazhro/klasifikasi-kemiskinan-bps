
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Menentukan lokasi proyek
ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "dataset_kemiskinan_bps.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Membaca dataset
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("EKSPLORASI DATASET KEMISKINAN BPS")
print("=" * 50)

# 3. Menampilkan ukuran dataset
print("\n1. UKURAN DATASET")
print("Jumlah baris:", df.shape[0])
print("Jumlah kolom:", df.shape[1])

# 4. Menampilkan 5 data pertama
print("\n2. LIMA DATA PERTAMA")
print(df.head())

# 5. Informasi tipe data
print("\n3. INFORMASI DATASET")
df.info()

# 6. Memeriksa nilai kosong
print("\n4. NILAI KOSONG")
print(df.isnull().sum())

# 7. Memeriksa data duplikat
print("\n5. DATA DUPLIKAT")
print("Duplikat seluruh baris:", df.duplicated().sum())
print(
    "Duplikat provinsi dan tahun:",
    df.duplicated(subset=["provinsi", "tahun"]).sum()
)

# 8. Statistik deskriptif
print("\n6. STATISTIK DESKRIPTIF")
print(df.describe())

# 9. Distribusi kategori kemiskinan
print("\n7. DISTRIBUSI KATEGORI")
print(df["kategori_kemiskinan"].value_counts())

# 10. Distribusi berdasarkan tahun
print("\n8. JUMLAH DATA PER TAHUN")
print(df["tahun"].value_counts().sort_index())

# 11. Memeriksa fitur dan target
FITUR = ["tahun", "ipm", "tpt_feb", "tpak_feb"]
TARGET = "kategori_kemiskinan"

print("\n9. FITUR MODEL")
print(FITUR)

print("\nTARGET MODEL")
print(TARGET)

# 12. Membuat visualisasi distribusi kategori
plt.figure(figsize=(8, 5))

df[TARGET].value_counts().reindex(
    ["Rendah", "Sedang", "Tinggi"]
).plot(kind="bar")

plt.title("Distribusi Kategori Kemiskinan")
plt.xlabel("Kategori Kemiskinan")
plt.ylabel("Jumlah Data")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distribusi_kategori.png",
    dpi=150
)

plt.close()

print("\nVisualisasi berhasil disimpan.")

# 13. Menyimpan ringkasan dataset
with open(
    OUTPUT_DIR / "ringkasan_dataset.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("RINGKASAN DATASET KEMISKINAN BPS\n")
    file.write(f"Jumlah data: {len(df)}\n")
    file.write(f"Jumlah kolom: {len(df.columns)}\n")
    file.write(f"Jumlah provinsi: {df['provinsi'].nunique()}\n")
    file.write("\nDistribusi kategori:\n")
    file.write(df[TARGET].value_counts().to_string())

print("\nEDA SELESAI!")

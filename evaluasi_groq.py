from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from groq import RateLimitError, APIError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

from classifier_groq import LABELS, make_client, classify_with_groq

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
TRAIN_PATH = OUTPUT_DIR / "train_data.csv"
TEST_PATH = OUTPUT_DIR / "test_data.csv"
EXAMPLES_PATH = MODEL_DIR / "llm_examples.csv"
PROGRESS_PATH = OUTPUT_DIR / "groq_progress.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "prediksi_groq_test.csv"

FEATURES = ["tahun", "ipm", "tpt_feb", "tpak_feb"]

def main():
    load_dotenv()

    for path in [TRAIN_PATH, TEST_PATH, EXAMPLES_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} belum ada. Jalankan python train_ml.py dahulu."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    examples = pd.read_csv(EXAMPLES_PATH)

    client = make_client()
    model_name = __import__("os").getenv(
        "GROQ_MODEL", "openai/gpt-oss-20b"
    )

    progress = {}
    if PROGRESS_PATH.exists():
        old = pd.read_csv(PROGRESS_PATH)
        progress = dict(zip(old["row_id"].astype(int), old["prediksi"]))

    print(f"Jumlah data test: {len(test_df)}")
    print(f"Prediksi yang sudah tersimpan: {len(progress)}")
    print(f"Model Groq: {model_name}")
    print("Satu data test = satu request API. Progres disimpan otomatis.")

    for _, row in test_df.iterrows():
        row_id = int(row["row_id"])
        if row_id in progress:
            continue

        try:
            label, raw_output = classify_with_groq(
                client, examples, row, model=model_name
            )
            progress[row_id] = label
            pd.DataFrame(
                [{"row_id": k, "prediksi": v} for k, v in progress.items()]
            ).to_csv(PROGRESS_PATH, index=False)

            print(
                f"Progres {len(progress)}/{len(test_df)} | "
                f"row_id={row_id} | prediksi={label}"
            )
            time.sleep(1.5)

        except RateLimitError as exc:
            print("\nGROQ RATE LIMIT.")
            print("Progres yang sudah selesai tetap tersimpan.")
            print("Jangan hapus groq_progress.csv.")
            print("Coba lanjutkan lagi setelah kuota tersedia.")
            print(str(exc))
            return

        except APIError as exc:
            print(f"\nAPI error pada row_id {row_id}: {exc}")
            print("Progres tersimpan. Jalankan ulang setelah masalah API selesai.")
            return

    pred_df = test_df[
        ["row_id", "tahun", "provinsi", "kategori_kemiskinan"]
    ].copy()
    pred_df["prediksi"] = pred_df["row_id"].map(progress)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    if pred_df["prediksi"].isna().any():
        print("Belum semua data berhasil diprediksi.")
        return

    y_true = pred_df["kategori_kemiskinan"].astype(str)
    y_pred = pred_df["prediksi"].astype(str)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true, y_pred, labels=LABELS,
        average="macro", zero_division=0
    )

    print("\n=== HASIL EVALUASI GROQ ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "n_test": int(len(pred_df)),
        "invalid": int((y_pred == "invalid").sum()),
    }
    with open(OUTPUT_DIR / "metrik_groq.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=LABELS
    )
    display.plot()
    plt.title("Confusion Matrix - Groq LLM")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_groq.png", dpi=160)
    plt.close()

    # Ringkasan perbandingan dua pendekatan pada test set yang sama.
    info_path = MODEL_DIR / "model_info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            ml_info = json.load(f)
        comparison = pd.DataFrame([
            {
                "metode": "Machine Learning - " + ml_info["model_name"],
                "accuracy": ml_info["accuracy"],
                "f1_macro": ml_info["f1_macro"],
            },
            {
                "metode": "AI Model Service - " + model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            },
        ])
        comparison.to_csv(
            OUTPUT_DIR / "perbandingan_ml_groq.csv", index=False
        )
        print("\nPerbandingan ML dan Groq:")
        print(comparison.to_string(index=False))

    print("\nHasil disimpan di outputs/prediksi_groq_test.csv")
    print("Metrik disimpan di outputs/metrik_groq.json")
    print("Confusion matrix disimpan di outputs/confusion_matrix_groq.png")

if __name__ == "__main__":
    main()

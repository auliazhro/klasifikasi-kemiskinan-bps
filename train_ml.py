from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset_kemiskinan_bps.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["tahun", "ipm", "tpt_feb", "tpak_feb"]
TARGET = "kategori_kemiskinan"
GROUP = "provinsi"
LABELS = ["Rendah", "Sedang", "Tinggi"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Pastikan dataset_kemiskinan_bps.csv berada di folder data."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET, GROUP]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan: {missing}")

    df = df.dropna(subset=required).copy()
    df[TARGET] = df[TARGET].astype(str)
    df[GROUP] = df[GROUP].astype(str)

    # ID baris dipertahankan agar prediksi dapat dilacak.
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    groups = df[GROUP].copy()

    print("=" * 65)
    print("STUDI KASUS: KLASIFIKASI TINGKAT KEMISKINAN PROVINSI")
    print("=" * 65)
    print(f"Jumlah data       : {len(df)}")
    print(f"Jumlah provinsi   : {groups.nunique()}")
    print("\nDistribusi kelas:")
    print(y.value_counts().reindex(LABELS, fill_value=0))

    # Lima fold, satu fold dipakai sebagai test.
    # StratifiedGroupKFold menjaga provinsi tidak tersebar ke train dan test.
    outer_cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    train_idx, test_idx = next(outer_cv.split(X, y, groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_df.to_csv(OUTPUT_DIR / "train_data.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_data.csv", index=False)

    print("\nPembagian data:")
    print(f"Train: {len(train_df)} baris, {train_df[GROUP].nunique()} provinsi")
    print(f"Test : {len(test_df)} baris, {test_df[GROUP].nunique()} provinsi")
    overlap = set(train_df[GROUP]) & set(test_df[GROUP])
    print(f"Provinsi yang muncul di train dan test: {len(overlap)}")
    if overlap:
        raise RuntimeError("Terjadi kebocoran kelompok provinsi.")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    groups_train = train_df[GROUP]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    inner_cv = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=42
    )

    comparison = []
    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            groups=groups_train,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        comparison.append({
            "model": name,
            "cv_f1_macro_mean": float(scores.mean()),
            "cv_f1_macro_std": float(scores.std()),
        })
        print(
            f"{name}: CV F1 Macro = {scores.mean():.4f} "
            f"(std {scores.std():.4f})"
        )

    comparison_df = pd.DataFrame(comparison).sort_values(
        "cv_f1_macro_mean", ascending=False
    )
    comparison_df.to_csv(OUTPUT_DIR / "perbandingan_model.csv", index=False)

    best_name = comparison_df.iloc[0]["model"]
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(
        y_test, predictions, labels=LABELS,
        average="macro", zero_division=0
    )

    print("\n" + "=" * 65)
    print(f"MODEL TERPILIH: {best_name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, labels=LABELS, zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    joblib.dump(best_model, MODEL_DIR / "model_kemiskinan.joblib")

    prediction_df = test_df[
        ["row_id", "tahun", "provinsi", TARGET]
    ].copy()
    prediction_df["prediksi"] = predictions
    prediction_df.to_csv(
        OUTPUT_DIR / "prediksi_ml_test.csv", index=False
    )

    cm = confusion_matrix(y_test, predictions, labels=LABELS)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=LABELS
    )
    display.plot()
    plt.title("Confusion Matrix - Machine Learning")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_ml.png", dpi=160)
    plt.close()

    # Contoh few-shot LLM diambil hanya dari data train, bukan data test.
    few_shot = (
        train_df.groupby(TARGET, group_keys=False)
        .sample(n=2, random_state=42)
        .reset_index(drop=True)
    )
    few_shot[FEATURES + [TARGET]].to_csv(
        MODEL_DIR / "llm_examples.csv", index=False
    )

    info = {
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "labels": LABELS,
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_provinces": int(train_df[GROUP].nunique()),
        "test_provinces": int(test_df[GROUP].nunique()),
        "split_method": "StratifiedGroupKFold; province groups kept separate",
        "note": (
            "persentase_miskin is used only to construct the target label "
            "and is excluded from model features."
        ),
    }
    with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\nFile tersimpan:")
    print("- models/model_kemiskinan.joblib")
    print("- models/model_info.json")
    print("- models/llm_examples.csv")
    print("- outputs/train_data.csv")
    print("- outputs/test_data.csv")
    print("- outputs/perbandingan_model.csv")
    print("- outputs/prediksi_ml_test.csv")
    print("- outputs/confusion_matrix_ml.png")
    print("\nSELESAI: model ML sudah dilatih dan dievaluasi.")

if __name__ == "__main__":
    main()

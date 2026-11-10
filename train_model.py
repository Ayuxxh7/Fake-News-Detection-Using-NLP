"""
train_model.py
--------------
Trains the Fake News Detection model and saves it.

Run:
    python train_model.py
    python train_model.py --data path/to/your_dataset.csv

The CSV must have two columns:
    text   -> the news headline / article
    label  -> FAKE or REAL  (0/1, true/false are also accepted)
Optionally a "title" column is joined with "text".

Outputs:
    model/trained_model.pkl            (vectorizer + trained model)
    model/metrics.json                 (measured evaluation results)
    documentation/confusion_matrix.png (confusion matrix picture)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI needed, just save the picture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "dataset" / "news.csv"
MODEL_PATH = BASE_DIR / "model" / "trained_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
CM_IMAGE_PATH = BASE_DIR / "documentation" / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def normalise_label(value):
    """Convert different label styles into the two strings FAKE / REAL."""
    v = str(value).strip().lower()
    if v in {"fake", "0", "false"}:
        return "FAKE"
    if v in {"real", "1", "true"}:
        return "REAL"
    return None  # unknown label


def load_dataset(path):
    """Read the CSV, check it, and return a clean DataFrame."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: dataset not found at {path}\n"
                 f"Run: python dataset/generate_sample_dataset.py   (or pass --data your.csv)")

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        sys.exit("ERROR: the CSV must contain the columns 'text' and 'label'. "
                 f"Found columns: {list(df.columns)}")

    if "title" in df.columns:
        df["text"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)

    df["label"] = df["label"].apply(normalise_label)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    counts = df["label"].value_counts()
    if len(counts) < 2 or counts.min() < 10:
        sys.exit("ERROR: need at least 10 examples of BOTH FAKE and REAL news to train.")
    return df


def save_confusion_matrix(cm, labels, path):
    """Draw the confusion matrix and save it as a PNG image."""
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the Fake News Detection model")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="path to the CSV dataset")
    args = parser.parse_args()

    # 1. Load data -------------------------------------------------------
    print(f"[1/6] Loading dataset: {args.data}")
    df = load_dataset(args.data)
    print(f"      Rows: {len(df)}  |  {df['label'].value_counts().to_dict()}")

    # 2. Preprocess ------------------------------------------------------
    print("[2/6] Preprocessing text (lowercase, punctuation, stopwords, tokenization)...")
    df["clean_text"] = df["text"].apply(preprocess_text)
    df = df[df["clean_text"].str.strip() != ""]

    # 3. Train / test split ---------------------------------------------
    print("[3/6] Splitting into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"],
    )

    # 4. TF-IDF + Logistic Regression -----------------------------------
    print("[4/6] Training TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)   # learn vocabulary ONLY from train data
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    # 5. Evaluation ------------------------------------------------------
    print("[5/6] Evaluating on the unseen test set...")
    y_pred = model.predict(X_test_tfidf)
    labels = ["FAKE", "REAL"]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="REAL", zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label="REAL", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="REAL", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Brier score checks how good the probabilities are (0 = perfect, 0.25 = guessing)
    p_real = model.predict_proba(X_test_tfidf)[:, list(model.classes_).index("REAL")]
    brier = brier_score_loss((y_test == "REAL").astype(int), p_real)

    # 5-fold cross validation (vectorizer is re-fit inside every fold -> no leakage)
    cv_pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    cv_scores = cross_val_score(
        cv_pipeline, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
    )

    # Most important words for each class (largest +/- coefficients)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]                      # classes_ = [FAKE, REAL]; positive -> REAL
    top_real = feature_names[np.argsort(coefs)[-10:][::-1]].tolist()
    top_fake = feature_names[np.argsort(coefs)[:10]].tolist()

    print("\n--------- RESULTS (positive class = REAL) ---------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 score  : {f1:.4f}")
    print(f"Brier     : {brier:.4f}")
    print(f"5-fold CV accuracy (train part): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print("Confusion matrix [rows = actual, cols = predicted] (FAKE, REAL):")
    print(cm)
    print("\n" + classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("Top words -> REAL:", top_real)
    print("Top words -> FAKE:", top_fake)

    # 6. Save ------------------------------------------------------------
    print("\n[6/6] Saving model and results...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_file": str(Path(args.data).name),
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "algorithm": "Logistic Regression",
        "vectorizer": "TF-IDF (max_features=5000, ngram_range=(1,2), min_df=2)",
        "positive_class": "REAL",
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "brier_score": round(float(brier), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "top_words_real": top_real,
        "top_words_fake": top_fake,
        "sklearn_version": sklearn.__version__,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(cm, labels, CM_IMAGE_PATH)

    print(f"Model saved   : {MODEL_PATH}")
    print(f"Metrics saved : {METRICS_PATH}")
    print(f"Matrix image  : {CM_IMAGE_PATH}")
    print("Done. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()

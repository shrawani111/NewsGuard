import os
import time
import joblib
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/train.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("   NEWSGUARD WORD + CHARACTER TF-IDF")
print("========================================\n")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["statement", "target"]
)

X_text = df["statement"].astype(str)
y = df["target"].astype(int)

print(f"Training samples: {len(df)}")
print(f"REAL samples: {(y == 1).sum()}")
print(f"FAKE samples: {(y == 0).sum()}")


# ============================================================
# WORD TF-IDF
# ============================================================

print("\nCreating word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_word = word_vectorizer.fit_transform(X_text)

print(f"Word TF-IDF shape: {X_word.shape}")


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print("\nCreating character TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=15000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_char = char_vectorizer.fit_transform(X_text)

print(f"Character TF-IDF shape: {X_char.shape}")


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\nCombining word + character features...")

X = hstack([
    X_word,
    X_char
]).tocsr()

print(f"Combined feature shape: {X.shape}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "SVM": LinearSVC(
        class_weight="balanced"
    )
}


# ============================================================
# STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []

best_f1 = 0
best_model_name = None


# ============================================================
# TRAINING
# ============================================================

for name, model in models.items():

    print("\n----------------------------------------")
    print(f"Training: {name}")
    print("----------------------------------------")

    start_time = time.time()

    fold_accuracy = []
    fold_precision = []
    fold_recall = []
    fold_f1 = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(f"Fold {fold}/5...")

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_val)

        fold_accuracy.append(
            accuracy_score(
                y_val,
                predictions
            )
        )

        fold_precision.append(
            precision_score(
                y_val,
                predictions,
                zero_division=0
            )
        )

        fold_recall.append(
            recall_score(
                y_val,
                predictions,
                zero_division=0
            )
        )

        fold_f1.append(
            f1_score(
                y_val,
                predictions,
                zero_division=0
            )
        )

    accuracy = sum(fold_accuracy) / len(fold_accuracy)
    precision = sum(fold_precision) / len(fold_precision)
    recall = sum(fold_recall) / len(fold_recall)
    f1 = sum(fold_f1) / len(fold_f1)

    elapsed = time.time() - start_time

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Time     : {elapsed:.2f} seconds")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Time_seconds": elapsed
    })

    if f1 > best_f1:

        best_f1 = f1
        best_model_name = name


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\n========================================")
print("          MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# TRAIN BEST MODEL ON COMPLETE DATASET
# ============================================================

print("\n========================================")
print(f"BEST MODEL: {best_model_name}")
print(f"CV F1 SCORE: {best_f1:.4f}")
print("========================================")


if best_model_name == "Logistic Regression":

    best_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

else:

    best_model = LinearSVC(
        class_weight="balanced"
    )


print("\nTraining best model on complete training data...")

best_model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL + VECTORIZERS
# ============================================================

joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "best_text_model.pkl"
    )
)

joblib.dump(
    word_vectorizer,
    os.path.join(
        MODEL_DIR,
        "word_tfidf.pkl"
    )
)

joblib.dump(
    char_vectorizer,
    os.path.join(
        MODEL_DIR,
        "char_tfidf.pkl"
    )
)

print("\nSaved:")

print(
    "models\\best_text_model.pkl"
)

print(
    "models\\word_tfidf.pkl"
)

print(
    "models\\char_tfidf.pkl"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "text_model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nResults saved to: {results_path}"
)


print("\n========================================")
print("      TEXT MODEL TRAINING COMPLETE")
print("========================================")
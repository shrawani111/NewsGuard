import os
import time
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from gensim.models import Word2Vec
import textstat
from textblob import TextBlob


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
print("     NEWSGUARD COMBINED FEATURES")
print("========================================\n")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["statement", "target"]
)

texts = df["statement"].astype(str)
y = df["target"].astype(int)

print(f"Training samples: {len(df)}")


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(texts)

print(f"TF-IDF shape: {X_tfidf.shape}")


# ============================================================
# WORD2VEC
# ============================================================

print("\nTraining Word2Vec...")

tokenized_texts = [
    text.split()
    for text in texts
]

word2vec = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10
)


def get_word2vec_vector(text):

    words = text.split()

    vectors = []

    for word in words:

        if word in word2vec.wv:
            vectors.append(
                word2vec.wv[word]
            )

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(
        vectors,
        axis=0
    )


print("Creating Word2Vec vectors...")

X_word2vec = np.array([
    get_word2vec_vector(text)
    for text in texts
])

print(
    f"Word2Vec shape: {X_word2vec.shape}"
)


# ============================================================
# READABILITY
# ============================================================

print("\nCreating readability features...")

readability = []

for text in texts:

    try:
        score = textstat.flesch_reading_ease(
            text
        )
    except Exception:
        score = 0

    readability.append(score)

X_readability = np.array(
    readability
).reshape(-1, 1)

print(
    f"Readability shape: {X_readability.shape}"
)


# ============================================================
# SENTIMENT
# ============================================================

print("\nCreating sentiment features...")

sentiment = []

for text in texts:

    try:
        score = TextBlob(
            text
        ).sentiment.polarity

    except Exception:
        score = 0

    sentiment.append(score)

X_sentiment = np.array(
    sentiment
).reshape(-1, 1)

print(
    f"Sentiment shape: {X_sentiment.shape}"
)


# ============================================================
# TEXT STATISTICS
# ============================================================

print("\nCreating text statistics...")

statistics = []

for text in texts:

    word_count = len(
        text.split()
    )

    character_count = len(
        text
    )

    punctuation_count = sum(
        1
        for char in text
        if not char.isalnum()
        and not char.isspace()
    )

    if character_count > 0:

        punctuation_ratio = (
            punctuation_count
            / character_count
        )

    else:

        punctuation_ratio = 0

    statistics.append([
        word_count,
        character_count,
        punctuation_ratio
    ])

X_statistics = np.array(
    statistics
)

print(
    f"Statistics shape: {X_statistics.shape}"
)


# ============================================================
# NORMALIZE NUMERIC FEATURES
# ============================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_word2vec_scaled = scaler.fit_transform(
    X_word2vec
)

X_readability_scaled = scaler.fit_transform(
    X_readability
)

X_sentiment_scaled = scaler.fit_transform(
    X_sentiment
)

X_statistics_scaled = scaler.fit_transform(
    X_statistics
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\nCombining all features...")

X_combined = hstack([
    X_tfidf,
    csr_matrix(X_word2vec_scaled),
    csr_matrix(X_readability_scaled),
    csr_matrix(X_sentiment_scaled),
    csr_matrix(X_statistics_scaled)
]).tocsr()

print(
    f"Combined feature shape: {X_combined.shape}"
)


# ============================================================
# MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


# ============================================================
# 5-FOLD STRATIFIED CROSS VALIDATION
# ============================================================

print("\n========================================")
print("       5-FOLD CROSS VALIDATION")
print("========================================")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_combined, y),
    start=1
):

    print(
        f"\nTraining fold {fold}/5..."
    )

    X_train = X_combined[train_idx]
    X_val = X_combined[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    accuracy_scores.append(
        accuracy_score(
            y_val,
            predictions
        )
    )

    precision_scores.append(
        precision_score(
            y_val,
            predictions,
            zero_division=0
        )
    )

    recall_scores.append(
        recall_score(
            y_val,
            predictions,
            zero_division=0
        )
    )

    f1_scores.append(
        f1_score(
            y_val,
            predictions,
            zero_division=0
        )
    )


# ============================================================
# RESULTS
# ============================================================

accuracy = np.mean(
    accuracy_scores
)

precision = np.mean(
    precision_scores
)

recall = np.mean(
    recall_scores
)

f1 = np.mean(
    f1_scores
)


print("\n========================================")
print("       COMBINED MODEL RESULTS")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining final model...")

final_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

final_model.fit(
    X_combined,
    y
)


# ============================================================
# SAVE EVERYTHING
# ============================================================

joblib.dump(
    final_model,
    os.path.join(
        MODEL_DIR,
        "combined_model.pkl"
    )
)

joblib.dump(
    tfidf,
    os.path.join(
        MODEL_DIR,
        "combined_tfidf.pkl"
    )
)

word2vec.save(
    os.path.join(
        MODEL_DIR,
        "word2vec.model"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "feature_scaler.pkl"
    )
)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}

pd.DataFrame(
    [metrics]
).to_csv(
    os.path.join(
        RESULTS_DIR,
        "combined_model_results.csv"
    ),
    index=False
)


print("\n========================================")
print("    COMBINED MODEL TRAINING COMPLETE")
print("========================================")

print("\nSaved files:")

print("models\\combined_model.pkl")
print("models\\combined_tfidf.pkl")
print("models\\word2vec.model")
print("models\\feature_scaler.pkl")
print("results\\combined_model_results.csv")
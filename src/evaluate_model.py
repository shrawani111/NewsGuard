import joblib
import pandas as pd

from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("       NEWSGUARD TEST EVALUATION")
print("========================================")

model = joblib.load(
    "models/best_text_model.pkl"
)

word_vectorizer = joblib.load(
    "models/word_tfidf.pkl"
)

char_vectorizer = joblib.load(
    "models/char_tfidf.pkl"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(
    "data/processed/test.csv"
)

test_df = test_df.dropna(
    subset=["statement", "target"]
)

X_text = test_df["statement"].astype(str)
y_test = test_df["target"].astype(int)


print(f"\nTest samples: {len(test_df)}")


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

print("\nTransforming test data...")

X_word_test = word_vectorizer.transform(
    X_text
)

X_char_test = char_vectorizer.transform(
    X_text
)

X_test = hstack([
    X_word_test,
    X_char_test
]).tocsr()


# ============================================================
# PREDICTIONS
# ============================================================

print("Making predictions...")

predictions = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\n========================================")
print("          TEST RESULTS")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["FAKE", "REAL"],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n========================================")
print("        CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_df["prediction"] = predictions

test_df["predicted_class"] = test_df[
    "prediction"
].map({
    0: "FAKE",
    1: "REAL"
})

test_df.to_csv(
    "results/test_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to: "
    "results/test_predictions.csv"
)


print("\n========================================")
print("       EVALUATION COMPLETE")
print("========================================")
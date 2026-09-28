import os
import joblib
import numpy as np
import shap

from scipy.sparse import hstack


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current file:
# C:\NewsGuard\src\explain.py

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_text_model.pkl"
)

WORD_TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "word_tfidf.pkl"
)

CHAR_TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "char_tfidf.pkl"
)


# ============================================================
# HEADER
# ============================================================

print("\n========================================")
print("       NEWSGUARD SHAP EXPLAINER")
print("========================================")


# ============================================================
# LOAD MODEL
# ============================================================

try:

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    word_vectorizer = joblib.load(
        WORD_TFIDF_PATH
    )

    char_vectorizer = joblib.load(
        CHAR_TFIDF_PATH
    )

    print("Model loaded successfully.")

except Exception as e:

    print("\nERROR: Could not load model files.")
    print(e)

    raise


# ============================================================
# FEATURE NAMES
# ============================================================

word_feature_names = (
    word_vectorizer
    .get_feature_names_out()
)

char_feature_names = (
    char_vectorizer
    .get_feature_names_out()
)


feature_names = np.concatenate([
    word_feature_names,
    char_feature_names
])


print(
    f"Total features: {len(feature_names)}"
)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

try:

    explainer = shap.LinearExplainer(
        model,
        shap.maskers.Independent(
            np.zeros(
                (
                    1,
                    len(feature_names)
                )
            )
        )
    )

    print("SHAP explainer ready.")

except Exception as e:

    print(
        "WARNING: SHAP explainer could not be created."
    )

    print(e)

    explainer = None


# ============================================================
# HELPER: GET SHAP VALUES
# ============================================================

def get_shap_values(X):

    if explainer is None:

        return None


    try:

        shap_result = explainer(X)

        values = np.asarray(
            shap_result.values
        )


        # --------------------------------------------
        # SHAP shape handling
        # --------------------------------------------

        if values.ndim == 3:

            values = values[
                0,
                :,
                0
            ]

        elif values.ndim == 2:

            values = values[0]

        elif values.ndim == 1:

            values = values

        else:

            values = values.flatten()


        return values


    except Exception as e:

        print(
            "\nSHAP calculation warning:"
        )

        print(e)

        return None


# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================

def explain_text(
    text,
    top_n=10
):

    # ========================================================
    # VALIDATE TEXT
    # ========================================================

    if not isinstance(
        text,
        str
    ):

        raise ValueError(
            "Text must be a string."
        )


    text = text.strip()


    if not text:

        raise ValueError(
            "Text cannot be empty."
        )


    # ========================================================
    # CREATE WORD TF-IDF
    # ========================================================

    word_x = (
        word_vectorizer.transform([
            text
        ])
    )


    # ========================================================
    # CREATE CHARACTER TF-IDF
    # ========================================================

    char_x = (
        char_vectorizer.transform([
            text
        ])
    )


    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    X = hstack([
        word_x,
        char_x
    ]).tocsr()


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = int(
        model.predict(X)[0]
    )


    if prediction == 1:

        predicted_class = "REAL"

    else:

        predicted_class = "FAKE"


    # ========================================================
    # PROBABILITIES
    # ========================================================

    fake_probability = 0.0
    real_probability = 0.0


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(X)[0]
        )


        # IMPORTANT:
        # Your model uses:
        # 0 = FAKE
        # 1 = REAL

        if len(probabilities) >= 2:

            fake_probability = (
                float(probabilities[0])
                * 100
            )

            real_probability = (
                float(probabilities[1])
                * 100
            )


        confidence = max(
            fake_probability,
            real_probability
        )


    else:

        decision = float(
            model.decision_function(X)[0]
        )


        probability = (
            1 /
            (
                1 +
                np.exp(-abs(decision))
            )
        )


        confidence = (
            probability * 100
        )


        if decision >= 0:

            real_probability = (
                probability * 100
            )

            fake_probability = (
                (1 - probability)
                * 100
            )

        else:

            fake_probability = (
                probability * 100
            )

            real_probability = (
                (1 - probability)
                * 100
            )


    # ========================================================
    # ACTIVE FEATURES
    # ========================================================

    # Get the actual columns that contain TF-IDF values.

    active_indices = X.nonzero()[1]

    active_indices = np.unique(
        active_indices
    )


    # ========================================================
    # SHAP VALUES
    # ========================================================

    shap_values = get_shap_values(
        X
    )


    signals = []


    # ========================================================
    # OPTION 1: SHAP EXPLANATION
    # ========================================================

    if shap_values is not None:

        for index in active_indices:

            index = int(index)


            if index >= len(
                feature_names
            ):

                continue


            feature = str(
                feature_names[index]
            ).strip()


            if not feature:

                continue


            # Ignore pure numbers

            if feature.isdigit():

                continue


            # Ignore one-character features

            if len(feature) <= 1:

                continue


            shap_value = float(
                shap_values[index]
            )


            signals.append({

                "feature":
                    feature,

                "shap":
                    round(
                        shap_value,
                        5
                    ),

                "direction":
                    (
                        "REAL"
                        if shap_value > 0
                        else "FAKE"
                    )

            })


    # ========================================================
    # SORT SHAP SIGNALS
    # ========================================================

    signals.sort(
        key=lambda item:
            abs(item["shap"]),
        reverse=True
    )


    # ========================================================
    # OPTION 2: LOGISTIC REGRESSION FALLBACK
    # ========================================================

    # If SHAP produced no usable signals, use the
    # Logistic Regression coefficients.

    if (
        len(signals) == 0
        and hasattr(model, "coef_")
    ):

        print(
            "\nUsing Logistic Regression "
            "feature contribution fallback..."
        )


        coefficients = (
            model.coef_[0]
        )


        # Only convert the one sample
        # to dense form.

        feature_values = (
            X.toarray()[0]
        )


        for index in active_indices:

            index = int(index)


            if index >= len(
                feature_names
            ):

                continue


            feature = str(
                feature_names[index]
            ).strip()


            if not feature:

                continue


            if feature.isdigit():

                continue


            if len(feature) <= 1:

                continue


            contribution = float(
                coefficients[index]
                * feature_values[index]
            )


            signals.append({

                "feature":
                    feature,

                "shap":
                    round(
                        contribution,
                        5
                    ),

                "direction":
                    (
                        "REAL"
                        if contribution > 0
                        else "FAKE"
                    )

            })


        signals.sort(
            key=lambda item:
                abs(item["shap"]),
            reverse=True
        )


    # ========================================================
    # REMOVE DUPLICATE FEATURES
    # ========================================================

    unique_signals = []

    seen = set()


    for signal in signals:

        feature = signal[
            "feature"
        ]


        if feature in seen:

            continue


        seen.add(
            feature
        )


        unique_signals.append(
            signal
        )


    signals = unique_signals


    # ========================================================
    # TOP SIGNALS
    # ========================================================

    signals = signals[
        :top_n
    ]


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n========================================")
    print("           PREDICTION")
    print("========================================")

    print(
        f"Prediction: {predicted_class}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        f"FAKE Probability: "
        f"{fake_probability:.2f}%"
    )

    print(
        f"REAL Probability: "
        f"{real_probability:.2f}%"
    )


    print("\n========================================")
    print("       TOP MODEL SIGNALS")
    print("========================================")


    if len(signals) == 0:

        print(
            "No model signals found."
        )

    else:

        for signal in signals:

            print(
                f"{signal['feature']:30} "
                f"{signal['shap']:+.5f} "
                f"→ {signal['direction']}"
            )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "prediction":
            predicted_class,

        "confidence":
            round(
                confidence,
                2
            ),

        "fake_probability":
            round(
                fake_probability,
                2
            ),

        "real_probability":
            round(
                real_probability,
                2
            ),

        "signals":
            signals

    }


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Scientists have discovered a revolutionary medicine
    that can cure every type of cancer within 24 hours.
    Doctors around the world are shocked by this discovery.
    """


    print("\n========================================")
    print("        RUNNING SHAP TEST")
    print("========================================")


    result = explain_text(
        sample_text,
        top_n=15
    )


    print("\n========================================")
    print("        SHAP TEST COMPLETE")
    print("========================================")
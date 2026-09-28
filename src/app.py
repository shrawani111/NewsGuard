import os
import joblib
import numpy as np

from flask import Flask, request, jsonify, render_template
from scipy.sparse import hstack

from explain import explain_text


# ============================================================
# PATH CONFIGURATION
# ============================================================

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

TEMPLATE_DIR = os.path.join(
    PROJECT_DIR,
    "templates"
)


# ============================================================
# MODEL PATHS
# ============================================================

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
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading NewsGuard model...")

try:

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

    print("ERROR loading model:")
    print(e)

    model = None
    word_vectorizer = None
    char_vectorizer = None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    if model is None:

        return jsonify({
            "status": "error",
            "model": "not loaded"
        }), 500

    return jsonify({

        "status": "ok",

        "application":
            "NewsGuard",

        "model":
            "loaded"

    })


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_news(text):

    if model is None:

        raise RuntimeError(
            "Model is not loaded."
        )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    word_features = (
        word_vectorizer.transform([
            text
        ])
    )


    char_features = (
        char_vectorizer.transform([
            text
        ])
    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = hstack([
        word_features,
        char_features
    ]).tocsr()


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        features
    )[0]


    if int(prediction) == 1:

        result = "REAL"

    else:

        result = "FAKE"


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                features
            )[0]
        )


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
            model.decision_function(
                features
            )[0]
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


    return {

        "prediction":
            result,

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
            )

    }


# ============================================================
# PREDICT API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success":
                    False,

                "error":
                    "No JSON data provided."

            }), 400


        text = data.get(
            "text",
            ""
        )


        if not isinstance(
            text,
            str
        ):

            return jsonify({

                "success":
                    False,

                "error":
                    "Text must be a string."

            }), 400


        text = text.strip()


        if not text:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please enter news text."

            }), 400


        if len(text) < 10:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please enter at least 10 characters."

            }), 400


        # ====================================================
        # NORMAL MODEL
        # ====================================================

        result = predict_news(
            text
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        print(
            "\nGenerating SHAP explanation..."
        )


        shap_result = explain_text(
            text,
            top_n=10
        )


        print(
            "SHAP signals:",
            len(
                shap_result.get(
                    "signals",
                    []
                )
            )
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "success":
                True,

            "text_length":
                len(text),

            "prediction":
                result[
                    "prediction"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "fake_probability":
                result[
                    "fake_probability"
                ],

            "real_probability":
                result[
                    "real_probability"
                ],

            "signals":
                shap_result.get(
                    "signals",
                    []
                )

        }


        print(
            "Returning signals:",
            response["signals"]
        )


        return jsonify(
            response
        )


    except Exception as e:

        print(
            "\nPrediction error:"
        )

        print(e)


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success":
            False,

        "error":
            "Page not found."

    }), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       NEWSGUARD API SERVER")
    print("========================================")

    print(
        "\nTemplate folder:"
    )

    print(
        TEMPLATE_DIR
    )

    print(
        "\nStarting server..."
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
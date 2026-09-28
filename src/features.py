import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import textstat
from textblob import TextBlob


# ============================================================
# TF-IDF FEATURES
# ============================================================

def create_tfidf(texts, max_features=5000):
    """
    Create TF-IDF features using unigrams and bigrams.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer


# ============================================================
# WORD2VEC FEATURES
# ============================================================

def train_word2vec(texts, vector_size=100):
    """
    Train a Word2Vec model on tokenized text.
    """

    tokenized_texts = [
        text.split()
        for text in texts
    ]

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=10
    )

    return model


def get_word2vec_vectors(texts, model):
    """
    Convert each text into an average Word2Vec sentence vector.
    """

    vectors = []

    for text in texts:

        words = text.split()

        word_vectors = []

        for word in words:

            if word in model.wv:
                word_vectors.append(model.wv[word])

        if word_vectors:
            sentence_vector = np.mean(
                word_vectors,
                axis=0
            )
        else:
            sentence_vector = np.zeros(
                model.vector_size
            )

        vectors.append(sentence_vector)

    return np.array(vectors)


# ============================================================
# READABILITY
# ============================================================

def readability_features(texts):
    """
    Calculate Flesch reading ease scores.
    """

    scores = []

    for text in texts:

        try:
            score = textstat.flesch_reading_ease(text)
        except Exception:
            score = 0

        scores.append(score)

    return np.array(scores).reshape(-1, 1)


# ============================================================
# SENTIMENT
# ============================================================

def sentiment_features(texts):
    """
    Calculate TextBlob sentiment polarity.
    """

    scores = []

    for text in texts:

        try:
            polarity = TextBlob(text).sentiment.polarity
        except Exception:
            polarity = 0

        scores.append(polarity)

    return np.array(scores).reshape(-1, 1)


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_statistics(original_texts):
    """
    Calculate article length and punctuation ratio.
    """

    features = []

    for text in original_texts:

        text = str(text)

        # Word count
        word_count = len(text.split())

        # Character count
        character_count = len(text)

        # Punctuation count
        punctuation_count = sum(
            1 for char in text
            if not char.isalnum() and not char.isspace()
        )

        # Punctuation ratio
        if character_count > 0:
            punctuation_ratio = (
                punctuation_count / character_count
            )
        else:
            punctuation_ratio = 0

        features.append([
            word_count,
            character_count,
            punctuation_ratio
        ])

    return np.array(features)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_texts = [
        "scientist discover new technology transform healthcare future",
        "government announce new economic policy today",
        "researchers develop innovative medical treatment"
    ]

    print("\n========================================")
    print("       NEWSGUARD FEATURE TEST")
    print("========================================")

    # TF-IDF
    tfidf_matrix, vectorizer = create_tfidf(sample_texts)

    print("\nTF-IDF shape:")
    print(tfidf_matrix.shape)

    # Word2Vec
    word2vec_model = train_word2vec(
        sample_texts,
        vector_size=50
    )

    word2vec_vectors = get_word2vec_vectors(
        sample_texts,
        word2vec_model
    )

    print("\nWord2Vec shape:")
    print(word2vec_vectors.shape)

    # Readability
    readability = readability_features(sample_texts)

    print("\nReadability shape:")
    print(readability.shape)

    # Sentiment
    sentiment = sentiment_features(sample_texts)

    print("\nSentiment shape:")
    print(sentiment.shape)

    # Text statistics
    statistics = text_statistics(sample_texts)

    print("\nText statistics shape:")
    print(statistics.shape)

    print("\n========================================")
    print("       FEATURE TEST COMPLETE")
    print("========================================")
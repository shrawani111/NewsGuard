import re
import spacy

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    """
    Clean and normalize a news statement.
    """

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text):
    """
    Complete NLP preprocessing:
    cleaning → tokenization → stopword removal → lemmatization
    """

    # Basic cleaning
    text = clean_text(text)

    # SpaCy processing
    doc = nlp(text)

    processed_tokens = []

    for token in doc:

        # Remove stopwords and punctuation
        if token.is_stop:
            continue

        if token.is_punct:
            continue

        # Remove very short tokens
        if len(token.text) < 2:
            continue

        # Lemmatization
        lemma = token.lemma_.lower().strip()

        if lemma:
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)


if __name__ == "__main__":

    sample_text = """
    Scientists have discovered a new technology that could
    transform healthcare in the future!
    Visit https://example.com for more information.
    """

    print("\nOriginal text:")
    print(sample_text)

    result = preprocess_text(sample_text)

    print("\nProcessed text:")
    print(result)
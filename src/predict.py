import re
import joblib


# Full article models
full_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
full_logistic_model = joblib.load("models/logistic_model.pkl")
full_svm_model = joblib.load("models/svm_calibrated_model.pkl")

# Headline models
headline_vectorizer = joblib.load("models/headline_vectorizer.pkl")
headline_logistic_model = joblib.load("models/headline_logistic_model.pkl")
headline_svm_model = joblib.load("models/headline_svm_model.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_news(text):
    cleaned = clean_text(text)

    # Short text = headline model
    # Long text = full article model
    word_count = len(cleaned.split())

    if word_count <= 40:
        vectorizer = headline_vectorizer
        logistic_model = headline_logistic_model
        svm_model = headline_svm_model
        model_type = "Headline Ensemble"
    else:
        vectorizer = full_vectorizer
        logistic_model = full_logistic_model
        svm_model = full_svm_model
        model_type = "Full-text Ensemble"

    features = vectorizer.transform([cleaned])

    logistic_probability = logistic_model.predict_proba(features)[0]
    svm_probability = svm_model.predict_proba(features)[0]

    ensemble_probability = (
        logistic_probability + svm_probability
    ) / 2

    fake_probability = float(ensemble_probability[0])
    real_probability = float(ensemble_probability[1])

    prediction = (
        "FAKE"
        if fake_probability > real_probability
        else "REAL"
    )

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "logistic_fake_probability": float(logistic_probability[0]),
        "svm_fake_probability": float(svm_probability[0]),
        "model_type": model_type,
        "word_count": word_count,
    }

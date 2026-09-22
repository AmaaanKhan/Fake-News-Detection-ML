import re
import joblib


# Load trained models
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
logistic_model = joblib.load("models/logistic_model.pkl")
svm_model = joblib.load("models/svm_calibrated_model.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_news(text):
    cleaned = clean_text(text)

    features = vectorizer.transform([cleaned])

    logistic_probability = logistic_model.predict_proba(features)[0]
    svm_probability = svm_model.predict_proba(features)[0]

    # Average the probabilities from both models
    ensemble_probability = (
        logistic_probability + svm_probability
    ) / 2

    fake_probability = ensemble_probability[0]
    real_probability = ensemble_probability[1]

    prediction = "FAKE" if fake_probability > real_probability else "REAL"

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "logistic_fake_probability": logistic_probability[0],
        "svm_fake_probability": svm_probability[0],
    }

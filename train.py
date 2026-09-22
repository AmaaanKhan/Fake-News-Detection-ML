import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load dataset
# -----------------------------

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)


# -----------------------------
# 2. Prepare text
# -----------------------------

data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


data["text"] = data["text"].apply(clean_text)


# -----------------------------
# 3. Train/test split
# -----------------------------

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# 4. TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# 5. Logistic Regression
# -----------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_tfidf, y_train)

logistic_pred = logistic_model.predict(X_test_tfidf)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, logistic_pred))
print(classification_report(y_test, logistic_pred))


# -----------------------------
# 6. SVM + probability calibration
# -----------------------------

svm_base = LinearSVC(random_state=42)

svm_model = CalibratedClassifierCV(
    svm_base,
    method="sigmoid",
    cv=3
)

svm_model.fit(X_train_tfidf, y_train)

svm_pred = svm_model.predict(X_test_tfidf)

print("\n=== SVM ===")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


# -----------------------------
# 7. Ensemble
# -----------------------------

logistic_prob = logistic_model.predict_proba(X_test_tfidf)
svm_prob = svm_model.predict_proba(X_test_tfidf)

ensemble_prob = (logistic_prob + svm_prob) / 2

ensemble_pred = ensemble_prob.argmax(axis=1)

print("\n=== Ensemble ===")
print("Accuracy:", accuracy_score(y_test, ensemble_pred))
print(classification_report(y_test, ensemble_pred))


# -----------------------------
# 8. Save models
# -----------------------------

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(logistic_model, "models/logistic_model.pkl")
joblib.dump(svm_model, "models/svm_calibrated_model.pkl")


print("\nAll models saved successfully.")

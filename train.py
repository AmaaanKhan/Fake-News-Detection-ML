import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score


DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("Loading dataset...")

fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

# ---------------------------------------------------------
# HEADLINE MODEL
# ---------------------------------------------------------

print("\nPreparing headline dataset...")

headlines = data[["title", "label"]].copy()
headlines["title"] = headlines["title"].fillna("").apply(clean_text)

headlines = headlines[headlines["title"].str.len() > 5]

X_headline = headlines["title"]
y_headline = headlines["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_headline,
    y_headline,
    test_size=0.20,
    random_state=42,
    stratify=y_headline
)

headline_vectorizer = TfidfVectorizer(
    max_features=30000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_features = headline_vectorizer.fit_transform(X_train)
X_test_features = headline_vectorizer.transform(X_test)

headline_logistic = LogisticRegression(
    max_iter=1000,
    random_state=42
)

headline_svm = CalibratedClassifierCV(
    LinearSVC(random_state=42),
    method="sigmoid",
    cv=3
)

print("Training headline Logistic Regression...")
headline_logistic.fit(X_train_features, y_train)

print("Training headline SVM...")
headline_svm.fit(X_train_features, y_train)

logistic_pred = headline_logistic.predict(X_test_features)
svm_pred = headline_svm.predict(X_test_features)

logistic_accuracy = accuracy_score(y_test, logistic_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

logistic_prob = headline_logistic.predict_proba(X_test_features)
svm_prob = headline_svm.predict_proba(X_test_features)

ensemble_prob = (logistic_prob + svm_prob) / 2
ensemble_pred = ensemble_prob.argmax(axis=1)

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print("\nHEADLINE MODEL RESULTS")
print("----------------------")
print(f"Logistic Regression: {logistic_accuracy:.4f}")
print(f"SVM:                 {svm_accuracy:.4f}")
print(f"Ensemble:            {ensemble_accuracy:.4f}")

# Save headline models
joblib.dump(
    headline_vectorizer,
    os.path.join(MODEL_DIR, "headline_vectorizer.pkl")
)

joblib.dump(
    headline_logistic,
    os.path.join(MODEL_DIR, "headline_logistic_model.pkl")
)

joblib.dump(
    headline_svm,
    os.path.join(MODEL_DIR, "headline_svm_model.pkl")
)

print("\nHeadline models saved.")


# ---------------------------------------------------------
# FULL TEXT MODEL
# ---------------------------------------------------------

print("\nPreparing full-text model...")

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["combined_text"] = (
    data["title"] + " " + data["text"]
).apply(clean_text)

X = data["combined_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

full_vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_features = full_vectorizer.fit_transform(X_train)
X_test_features = full_vectorizer.transform(X_test)

full_logistic = LogisticRegression(
    max_iter=1000,
    random_state=42
)

full_svm = CalibratedClassifierCV(
    LinearSVC(random_state=42),
    method="sigmoid",
    cv=3
)

print("Training full-text Logistic Regression...")
full_logistic.fit(X_train_features, y_train)

print("Training full-text SVM...")
full_svm.fit(X_train_features, y_train)

logistic_pred = full_logistic.predict(X_test_features)
svm_pred = full_svm.predict(X_test_features)

logistic_accuracy = accuracy_score(y_test, logistic_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

logistic_prob = full_logistic.predict_proba(X_test_features)
svm_prob = full_svm.predict_proba(X_test_features)

ensemble_prob = (logistic_prob + svm_prob) / 2
ensemble_pred = ensemble_prob.argmax(axis=1)

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print("\nFULL-TEXT MODEL RESULTS")
print("----------------------")
print(f"Logistic Regression: {logistic_accuracy:.4f}")
print(f"SVM:                 {svm_accuracy:.4f}")
print(f"Ensemble:            {ensemble_accuracy:.4f}")

# Save full-text models
joblib.dump(
    full_vectorizer,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

joblib.dump(
    full_logistic,
    os.path.join(MODEL_DIR, "logistic_model.pkl")
)

joblib.dump(
    full_svm,
    os.path.join(MODEL_DIR, "svm_calibrated_model.pkl")
)

print("\nFull-text models saved.")
print("\nTRAINING COMPLETE.")

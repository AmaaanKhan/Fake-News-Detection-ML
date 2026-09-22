# Fake News Detection using Machine Learning

A machine learning-based fake news classification system that accepts either text or a newspaper/news image. Images are processed using OCR before being passed to the machine learning models.

## Features

- Text-based news classification
- Newspaper/news image input
- OCR using Tesseract
- Headline-specific classification model
- Full-text classification model
- TF-IDF text feature extraction
- Logistic Regression
- Support Vector Machine (SVM)
- Ensemble prediction using Logistic Regression and SVM
- Fake and real probability scores
- Streamlit web interface

### Text Input

Text → Preprocessing → TF-IDF → Logistic Regression + SVM → Ensemble → Prediction

### Newspaper Image Input

Newspaper Image → Tesseract OCR → Extracted Text → Preprocessing → Headline/Full-text Model → Ensemble → Prediction

## Machine Learning Models

The project uses two classification pipelines.

### Headline Model

Used for short inputs such as news headlines and OCR-extracted newspaper headlines.

- TF-IDF Vectorizer
- Logistic Regression
- Calibrated Linear SVM
- Ensemble of both models

Headline test accuracy:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 94.90% |
| SVM | 95.90% |
| Ensemble | 95.69% |

### Full-text Model

Used for longer news articles.

- TF-IDF Vectorizer
- Logistic Regression
- Calibrated Linear SVM
- Ensemble of both models

Full-text test accuracy:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 99.22% |
| SVM | 99.74% |
| Ensemble | 99.57% |

## Dataset

The project uses the Fake and Real News Dataset containing separate fake and real news articles.

The dataset contains:

- `Fake.csv`
- `True.csv`

Each file contains:

- `title`
- `text`
- `subject`
- `date`

The dataset is kept locally and excluded from Git using `.gitignore`.

## OCR

Tesseract OCR is used to extract text from uploaded newspaper images.

OCR is treated as the input-processing stage rather than as the classification model.

## Prediction

The system automatically selects the appropriate model based on input length.

- Short input → Headline Ensemble
- Long input → Full-text Ensemble

The system reports:

- Prediction
- Fake probability
- Real probability
- Model used

The probability represents the model's classification confidence based on patterns learned from the training dataset. It does not independently verify whether a news claim is factually true.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Tesseract OCR
- PyTesseract
- Pillow
- Streamlit
- Git/GitHub


## Screenshots
<img width="1917" height="1017" alt="Screenshot 2026-09-22 181024" src="https://github.com/user-attachments/assets/cf8fa873-ab74-464e-8984-5b7e6f60461f" />
<img width="1910" height="1018" alt="Screenshot 2026-09-22 181038" src="https://github.com/user-attachments/assets/9fb1a332-7e38-4005-a84c-0cc551329b45" />
<img width="1917" height="1017" alt="Screenshot 2026-09-22 184726" src="https://github.com/user-attachments/assets/4cf06917-3da3-43a9-8ce2-6b4333ed2804" />
<img width="1917" height="1017" alt="Screenshot 2026-09-22 184745" src="https://github.com/user-attachments/assets/7398868a-8418-4515-b34e-f6bf70c48e1a" />



## Project Structure

```text
Fake-News-Detection-ML/
├── data/
├── models/
├── src/
│   ├── ocr.py
│   └── predict.py
├── tests/
├── screenshots/
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md

import streamlit as st
from src.predict import predict_news
from src.ocr import extract_text_from_image

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.title("Fake News Detection using Machine Learning")
st.write(
    "Detect whether a news headline or article is likely to be fake or real "
    "using TF-IDF, Logistic Regression, and Support Vector Machine."
)

st.divider()

input_mode = st.radio(
    "Choose input method",
    ["Text Input", "Newspaper / News Image"],
    horizontal=True
)

if input_mode == "Text Input":

    text = st.text_area(
        "Enter a news headline or article",
        height=220,
        placeholder="Paste news text here..."
    )

    if st.button("Analyze News", type="primary"):

        if not text.strip():
            st.warning("Please enter some news text.")
        else:
            result = predict_news(text)

            st.subheader("Prediction")

            if result["prediction"] == "FAKE":
                st.error("FAKE NEWS")
            else:
                st.success("REAL NEWS")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Fake Probability",
                    f"{result['fake_probability'] * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Real Probability",
                    f"{result['real_probability'] * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Words",
                    result["word_count"]
                )

            st.info(
                f"Model used: **{result['model_type']}**"
            )

elif input_mode == "Newspaper / News Image":

    uploaded_file = st.file_uploader(
        "Upload a newspaper or news image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded news image",
            use_container_width=True
        )

        if st.button("Extract & Analyze", type="primary"):

            with st.spinner("Extracting text using OCR..."):

                extracted_text = extract_text_from_image(uploaded_file)

            if not extracted_text:
                st.warning("No text could be extracted from the image.")
            else:

                st.subheader("Extracted Text")

                st.text_area(
                    "OCR Output",
                    extracted_text,
                    height=200
                )

                result = predict_news(extracted_text)

                st.subheader("Prediction")

                if result["prediction"] == "FAKE":
                    st.error("FAKE NEWS")
                else:
                    st.success("REAL NEWS")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Fake Probability",
                        f"{result['fake_probability'] * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Real Probability",
                        f"{result['real_probability'] * 100:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Words",
                        result["word_count"]
                    )

                st.info(
                    f"Model used: **{result['model_type']}**"
                )

st.divider()

st.caption(
    "Machine-learning classification based on patterns learned from the training dataset. "
    "The result does not independently verify whether a news claim is factually true."
)

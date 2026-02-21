import streamlit as st
import joblib
import pandas as pd
import os

# ----------------------------
# Load model and vectorizer safely
# ----------------------------
MODEL_PATH = "svm_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("svm_model.pkl not found. Make sure it is in the project folder.")
    st.stop()

if not os.path.exists(VECTORIZER_PATH):
    st.error("vectorizer.pkl not found. Make sure it is in the project folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ----------------------------
# Page settings
# ----------------------------
st.set_page_config(
    page_title="Healthcare Sentiment Analyzer",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare Review Sentiment Analyzer")
st.write("Enter a healthcare review to predict sentiment.")

# ----------------------------
# History storage
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

review = st.text_area("Patient Review", height=150)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        vector = vectorizer.transform([review])
        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = round(probabilities[class_index] * 100, 2)

        if prediction.lower() == "positive":
            st.success(f"Predicted Sentiment: {prediction}")
        elif prediction.lower() == "negative":
            st.error(f"Predicted Sentiment: {prediction}")
        else:
            st.info(f"Predicted Sentiment: {prediction}")

        st.write(f"Confidence: {confidence}%")

        st.session_state.history.append({
            "Review": review,
            "Prediction": prediction,
            "Confidence (%)": confidence
        })

if st.session_state.history:
    st.markdown("---")
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

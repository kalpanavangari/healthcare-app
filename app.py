import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Healthcare Sentiment Analyzer",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare Review Sentiment Analyzer (SVM)")
st.markdown("Enter a healthcare review below to predict sentiment.")

# -----------------------------
# Session History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Input Box
# -----------------------------
review = st.text_area("✍️ Patient Review", height=150)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Transform text
        vector = vectorizer.transform([review])

        # Predict sentiment
        prediction = model.predict(vector)[0]

        # Confidence Score (SVM)
        decision = model.decision_function(vector)

        if len(model.classes_) == 2:
            confidence = abs(decision[0])
        else:
            confidence = np.max(decision)

        confidence_percent = round(float(abs(confidence)) * 100, 2)

        # Display Result
        if prediction.lower() == "positive":
            st.success(f"Predicted Sentiment: {prediction}")
        elif prediction.lower() == "negative":
            st.error(f"Predicted Sentiment: {prediction}")
        else:
            st.info(f"Predicted Sentiment: {prediction}")

        st.write(f"Confidence Score: {confidence_percent}%")

        # Save to history
        st.session_state.history.append({
            "Review": review,
            "Prediction": prediction,
            "Confidence (%)": confidence_percent
        })

# -----------------------------
# Show History
# -----------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

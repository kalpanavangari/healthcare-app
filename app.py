import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Load model and vectorizer
# ----------------------------
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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

# ----------------------------
# Input
# ----------------------------
review = st.text_area("Patient Review", height=150)

# ----------------------------
# Predict
# ----------------------------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Convert text to vector
        vector = vectorizer.transform([review])

        # Predict sentiment
        prediction = model.predict(vector)[0]

        # Get probabilities
        probabilities = model.predict_proba(vector)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = round(probabilities[class_index] * 100, 2)

        # Display result
        if prediction.lower() == "positive":
            st.success(f"Predicted Sentiment: {prediction}")
        elif prediction.lower() == "negative":
            st.error(f"Predicted Sentiment: {prediction}")
        else:
            st.info(f"Predicted Sentiment: {prediction}")

        st.write(f"Confidence: {confidence}%")

        # Save history
        st.session_state.history.append({
            "Review": review,
            "Prediction": prediction,
            "Confidence (%)": confidence
        })

# ----------------------------
# Show history
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

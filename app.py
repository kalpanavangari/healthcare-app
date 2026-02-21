import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Healthcare Sentiment AI", layout="centered")

# Custom CSS for premium look
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
        }
        .title {
            font-size:40px;
            font-weight:bold;
            text-align:center;
            color:#4CAF50;
        }
        .subtitle {
            text-align:center;
            color:gray;
            margin-bottom:30px;
        }
        .result-box {
            padding:20px;
            border-radius:10px;
            font-size:20px;
            font-weight:bold;
            text-align:center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🏥 Healthcare Review Sentiment AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">SVM-based Intelligent Sentiment Classification System</div>', unsafe_allow_html=True)

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Input
review = st.text_area("Enter Patient Review", height=150)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        vector = vectorizer.transform([review])
        prediction = model.predict(vector)[0]

        # Confidence (SVM decision function)
        confidence = model.decision_function(vector)
        confidence_score = round(abs(confidence[0]), 3)

        # If probability available
        try:
            probs = model.predict_proba(vector)[0]
            probability = round(max(probs) * 100, 2)
        except:
            probability = None

        # Result styling
        if prediction == "positive":
            st.success(f"Predicted Sentiment: POSITIVE 😊")
        elif prediction == "negative":
            st.error(f"Predicted Sentiment: NEGATIVE 😡")
        else:
            st.info(f"Predicted Sentiment: NEUTRAL 😐")

        st.write(f"Confidence Score: {confidence_score}")

        if probability:
            st.write(f"Probability: {probability}%")

        # Save to history
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Review": review,
            "Prediction": prediction
        })

# Show history
if st.session_state.history:
    st.markdown("### 📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

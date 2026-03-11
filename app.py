import streamlit as st
import joblib
from preprocess import preprocess_text

# Load model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("📧 Spam Email Classifier")

st.write("Enter an email message to check whether it is spam or not.")

email_text = st.text_area("Enter Email Message")

if st.button("Predict"):

    processed = preprocess_text(email_text)

    vector = vectorizer.transform([processed]).toarray()

    prediction = model.predict(vector)

    if prediction[0] == 0:
        st.error("🚨 This is a Spam Email")

    else:
        st.success("✅ This is a Normal Email")
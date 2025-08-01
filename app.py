import os
import gdown
import zipfile

# Google Drive File ID of your ZIP
FILE_ID = "1grtTJZ30zJV8pr8Xvy-lPZ6xeodzgbdp"  # Replace this with your file ID
ZIP_PATH = "model_files.zip"
EXTRACT_DIR = "model_files"

# Download ZIP if not already downloaded
if not os.path.exists(ZIP_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)

# Extract ZIP if not already extracted
if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# Load model and vectorizer from extracted folder
import pickle
model = pickle.load(open(os.path.join(EXTRACT_DIR, 'trained_model.sav'), 'rb'))
vectorizer = pickle.load(open(os.path.join(EXTRACT_DIR, 'vectorizer.sav'), 'rb'))

import streamlit as st
#import pickle
import numpy as np




# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Page logic
if st.session_state["page"] == "home":
    # Title
    st.title("🐦 Twitter Sentiment Analysis")
    st.write("""
    This project predicts whether a tweet expresses a **Positive** or **Negative** sentiment
    using a Logistic Regression model trained on Twitter data.
    """)

    # Features
    st.markdown("### Features:")
    st.markdown("""
    - Real-time sentiment prediction
    - Confidence score display
    - Clean and simple user interface
    """)

    # Button to navigate
    if st.button("Go to Sentiment Analysis →"):
        st.session_state["page"] = "analysis"


elif st.session_state["page"] == "analysis":
    # Back button to return to home
    if st.button("← Back to Home"):
        st.session_state["page"] = "home"

    # Title for analysis page
    st.header("Analyze Tweet Sentiment")

    # Text input
    user_input = st.text_input("Enter a tweet to analyze:")

    # Analyze button (for now just placeholder)
    if st.button("Analyze"):
        if user_input.strip() == "":
            st.error("Please enter a tweet before analyzing.")
        else:
            processed_text = vectorizer.transform([user_input])  # Convert text to features

            # Prediction
            prediction = model.predict(processed_text)
            confidence = model.predict_proba(processed_text)

            # Get confidence for predicted class
            confidence_score = np.max(confidence) * 100  # in percentage

            # Display sentiment
            if prediction[0] == 1:
                st.success(f"**Positive Sentiment 😀** (Confidence: {confidence_score:.2f}%)")
            else:
                st.error(f"**Negative Sentiment 😠** (Confidence: {confidence_score:.2f}%)")

    st.markdown("### Or upload a CSV file to analyze multiple tweets")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    import pandas as pd

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
    
    #if uploaded_file is not None:
        if st.button("Analyze CSV"):
            tweets = df['tweet'].astype(str)  # Ensure string type
            transformed = vectorizer.transform(tweets)

            preds = model.predict(transformed)
            confs = model.predict_proba(transformed)

            # Build result DataFrame
            results = pd.DataFrame({
                'Tweet': tweets,
                'Sentiment': ['Positive' if p == 1 else 'Negative' for p in preds],
                'Confidence (%)': [f"{max(c)*100:.2f}" for c in confs]
            })

            st.write("Analysis Results:")
            st.dataframe(results)
        sentiment_counts = results['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

# Sidebar Content
st.sidebar.title("About Project")

st.sidebar.write("""
**Twitter Sentiment Analysis**

This app analyzes tweets to determine whether they are **Positive** or **Negative** 
using a Logistic Regression model trained on Twitter data.
""")

st.sidebar.markdown("---")
st.sidebar.write("**Created by:** Yash Raj")  # Your Name

# Links
st.sidebar.markdown("[GitHub](https://github.com/Yash-Raj1)")
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/yash-raj04)")

# Model Info
st.sidebar.markdown("---")
st.sidebar.write("**Model Accuracy:** ~85% Training, 79% Testing")



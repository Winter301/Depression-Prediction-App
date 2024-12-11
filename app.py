import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required libraries are installed
try:
    import pandas
    import matplotlib
    import seaborn
    import sklearn
    import wordcloud
except ImportError:
    raise ImportError("Make sure to install required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, wordcloud")

# Load the pre-trained model and vectorizer
with open('best_svc_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app title
st.title("Depression Prediction App")

# Instructions for the user
st.write("This app uses a machine learning model to predict if a given text indicates depression. Please enter a sentence or text in the input box below.")

# User input for text
user_input = st.text_area("Enter your sentence or words:", height=150)

# Prediction button
if st.button("Predict"):
    if user_input.strip():
        # Transform the input using the loaded vectorizer
        input_vectorized = vectorizer.transform([user_input]).toarray()
        
        # Make prediction
        prediction = model.predict(input_vectorized)
        
        # Display the result
        if prediction[0] == 1:
            st.error("The model predicts: You may be experiencing depression. Please consider reaching out to a mental health professional for support.")
        else:
            st.success("The model predicts: You are not experiencing depression. Stay positive and take care of your mental health!")
    else:
        st.warning("Please enter some text for prediction.")

# Footer
st.write("\n---\n")
st.caption("Disclaimer: This app is for informational purposes only and is not a substitute for professional mental health advice. If you are feeling distressed, please seek help from a licensed professional.")

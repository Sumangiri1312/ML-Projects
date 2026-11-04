
import streamlit as st
import pickle
import string
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open(BASE_DIR / 'model.pkl', 'rb') as file:
    model = pickle.load(file)


# Streamlit interface
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if not input_sms.strip():
        st.warning("Please enter a message.")

    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")

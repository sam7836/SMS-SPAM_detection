import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')


# Download required NLTK data only once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [word for word in text if word.isalnum()]
    y = [word for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    y = [ps.stem(word) for word in y]

    return " ".join(y)

# Load vectorizer and model
@st.cache_resource  # Caches the model to avoid reloading on each run
def load_resources():
    vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
    return vectorizer, model

vectorizer, model = load_resources()

# Streamlit app title
st.title("📩 SMS Spam Detection Model")

# Input field for SMS
st.write("Enter the SMS text to predict if it's spam or not:")
input_sms = st.text_input("")

if st.button('Predict'):
    if not input_sms.strip():
        st.error("Please enter a valid SMS text.")
    else:
        # Preprocess and predict
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display result with styling
        if result == 1:
            st.error("🛑 This SMS is classified as **Spam**!")
        else:
            st.success("✅ This SMS is classified as **Not Spam**.")

# Sidebar with information
st.sidebar.title("About the App")
st.sidebar.info(
    """
    - **Spam Detection Model** trained on a dataset of SMS messages.
    - Uses Natural Language Processing (NLP) techniques for text preprocessing.
    """
)

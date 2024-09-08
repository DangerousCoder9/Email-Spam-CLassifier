from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import logging
import os

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Porter Stemmer
pt = PorterStemmer()

app = Flask(__name__)

# Load the model and the TF-IDF vectorizer using absolute paths
model_path = os.path.join(os.getcwd(), 'model/tfidf_ann_model.h5')
vectorizer_path = os.path.join(os.getcwd(), 'model/tfidf_vectorizer.pkl')

try:
    model = tf.keras.models.load_model(model_path)
    with open(vectorizer_path, 'rb') as file:
        tfidf = pickle.load(file)
    app.logger.info("Model and vectorizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model or vectorizer: {e}")
    raise

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    res = [i for i in text if i.isalnum()]
    res = [pt.stem(i) for i in res if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(res)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input
        message = request.form['message']
        
        # Transform the message
        transformed_message = transform_text(message)
        
        # Vectorize the transformed message
        data = [transformed_message]
        vect = tfidf.transform(data).toarray()
        
        # Make prediction
        prediction = model.predict(vect)
        
        # Interpret the prediction
        predicted_label = 'Spam' if prediction[0] > 0.5 else 'Ham'
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        predicted_label = "Error in prediction"
    
    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    # Set up logging for debugging purposes
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=True)

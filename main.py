import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

#Load The IMDB dataset word Index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#Load the pretrained model with relu activation

model = load_model('simple_rnn_imdb.h5')


#Function to preprocess user input 

# Helper Functions
# Function to decode reviews

def decode_review(encoded_reviews):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_reviews])

#Function to preprocess user input

def preprocess_text(text):
    words = text.lower().split()
    encoded_reviews = [word_index.get(word, 2) + 3 for word in words]
    padded_reviews = sequence.pad_sequences([encoded_reviews], maxlen = 500)
    return padded_reviews



## Prediction Function


def predict_sentiment(review):
    presprocessed_input = preprocess_text(review)

    prediction = model.predict(presprocessed_input)

    sentiment = 'Postive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


## STream lit app

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

#user input

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    presprocess_input = preprocess_text(user_input)
    
    #Make Prdiction

    prediction = model.predict(presprocess_input)
    
    sentiment = 'Postive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")

else:
    st.write('Please Enter a Movie Review')

    

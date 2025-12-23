# Sentiment-Analysis-using-Simple-RNN
This project is a web-based sentiment analysis application that classifies IMDB movie reviews as Positive or Negative using a Simple Recurrent Neural Network (RNN) built with TensorFlow/Keras. The application provides an interactive interface using Streamlit, allowing users to input custom movie reviews and receive real-time sentiment predictions along with confidence scores.

The model is trained on the IMDB Movie Reviews dataset, where text reviews are converted into numerical sequences using the official IMDB word index and padded to a fixed length before being passed to the RNN model.

# Key Features

Simple RNN-based text classification model

IMDB dataset word indexing and sequence padding

Real-time sentiment prediction via Streamlit UI

Displays prediction confidence score

End-to-end NLP pipeline (preprocessing → prediction → visualization)

# ⚠️Model Limitations (Important Learning Insight)

This project intentionally uses a Simple RNN, which has known limitations in semantic understanding. The model may misclassify short or negated sentences (e.g., “It was a bad movie”) because RNNs rely on sequential statistical patterns rather than true language semantics. These limitations highlight why modern NLP systems prefer LSTMs and Transformer-based models (BERT, GPT) for sentiment analysis.

# Purpose of the Project

To understand how sequential neural networks process text

To explore real-world limitations of RNNs in NLP tasks

To build and deploy an interactive ML application

To serve as a foundational project before advancing to LSTM and Transformer models

# 🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy

Streamlit

IMDB Movie Reviews Dataset

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle  # If tokenizer was saved separately

# Load the trained model
model = tf.keras.models.load_model(r"E:\Projects\GRU Q and A bot\seq2seq_model_gru.h5")

# Load the tokenizer (if saved separately)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 19  # Same as used during training
VOCAB_SIZE = 10000  # Adjust if needed

def generate_response(user_input):
    # Preprocess input
    user_seq = tokenizer.texts_to_sequences([user_input])
    user_seq_padded = pad_sequences(user_seq, maxlen=MAX_LENGTH, padding='post')

    # Generate prediction
    predicted_seq = model.predict([user_seq_padded, np.zeros_like(user_seq_padded)])

    # Convert prediction to text
    predicted_indices = np.argmax(predicted_seq, axis=-1)
    predicted_text = tokenizer.sequences_to_texts(predicted_indices)[0]

    return predicted_text.replace("<start>", "").replace("<end>", "").strip()

st.title("Chatbot - Trained on 10,000 Entries")

user_input = st.text_input("You:", "")

if st.button("Send"):
    response = generate_response(user_input)
    st.text_area("Bot:", response, height=100)

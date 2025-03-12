import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model(r"E:\Projects\GRU Q and A bot\seq2seq_model_gru.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 19  # Same as used during training
VOCAB_SIZE = 10000  # Adjust if needed

def generate_response(user_input):
    # Preprocess input: shape => (1, 19)
    user_seq = tokenizer.texts_to_sequences([user_input])
    user_seq_padded = pad_sequences(user_seq, maxlen=MAX_LENGTH, padding='post')

    # The decoder input should have shape => (1, 18)
    decoder_placeholder = np.zeros((1, MAX_LENGTH - 1), dtype=user_seq_padded.dtype)

    # Generate prediction
    # Pass the encoder input (1, 19) and a dummy decoder input (1, 18)
    predicted_seq = model.predict([user_seq_padded, decoder_placeholder])

    # Convert prediction to text
    predicted_indices = np.argmax(predicted_seq, axis=-1)
    predicted_text = tokenizer.sequences_to_texts(predicted_indices)[0]

    # Remove any <start>/<end> tokens, if present
    return predicted_text.replace("<start>", "").replace("<end>", "").strip()

st.title("Chatbot - Trained on 10,000 Entries")

user_input = st.text_input("You:", "")

if st.button("Send"):
    response = generate_response(user_input)
    st.text_area("Bot:", response, height=100)

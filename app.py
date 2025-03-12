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


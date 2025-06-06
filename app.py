import streamlit as st

# ✅ Set page config FIRST
st.set_page_config(page_title="Emotion Classifier", page_icon="💬")

# Now continue with other imports
import numpy as np
import pickle
import sys
import tensorflow.keras as tf_keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GRU
from tensorflow.keras.saving import register_keras_serializable

# ✅ Patch keras module paths
sys.modules['keras.preprocessing.text'] = tf_keras.preprocessing.text
sys.modules['keras.preprocessing.sequence'] = tf_keras.preprocessing.sequence

# ✅ Custom GRU layer registration
@register_keras_serializable()
class CustomGRU(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# ✅ Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("emotion_model.h5", custom_objects={'GRU': CustomGRU})
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Class labels and maxlen
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
maxlen = 50  # Must match model input length

# Streamlit UI
st.title("💬Emotion Classification model")
st.markdown("Enter a tweet or sentence, and this app will predict the emotion behind it.")

user_input = st.text_area("✍️ Your Input:")

if st.button("🔍 Predict Emotion"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_input = pad_sequences(sequence, maxlen=maxlen)
        prediction = model.predict(padded_input)
        predicted_emotion = labels[np.argmax(prediction)]
        st.success(f"🎯 **Predicted Emotion:** {predicted_emotion.capitalize()}")

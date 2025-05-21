import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

@st.cache_resource(show_spinner=False)
def load_lstm_model_and_tokenizer():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_lstm_model_and_tokenizer()

max_len = 100
labels_map = {
    0: "😠 سلبي",
    1: "😐 محايد",
    2: "😄 إيجابي"
}

st.title("تصنيف المشاعر باستخدام LSTM")

user_input = st.text_area("أدخلي نص التقييم هنا:")

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)
    label = np.argmax(pred, axis=1)[0]
    return label

if st.button("تحليل المشاعر"):
    if user_input.strip() == "":
        st.warning("الرجاء إدخال نص.")
    else:
        prediction = predict_sentiment(user_input)
        st.success(f"التصنيف: {labels_map[prediction]}")

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import random

# Set repo model kamu di HF
MODEL_ID = "athayary/best_model_cendol"  # Ganti dengan repo kamu

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@st.cache_data
def load_intents():
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    responses = {}
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']
    return responses

tokenizer, model, device = load_model()
intent_responses = load_intents()

# Prediction function
def predict_intent(text):
    input_text = "Tentukan intent dari kalimat berikut: " + text.lower().strip()
    enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_length=8)

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return pred

def get_response(intent):
    if intent in intent_responses:
        return random.choice(intent_responses[intent])
    else:
        return "Maaf, saya tidak mengerti maksud Anda."

# Streamlit UI
st.title("🤖 TanyaRasa Chatbot")
st.write("Chatbot berbasis model fine-tuned IndoNLP CENDOL")

user_input = st.text_input("Masukkan pertanyaan Anda:")
if st.button("Tanya"):
    if user_input:
        intent = predict_intent(user_input)
        response = get_response(intent)
        st.markdown(f"**Intent:** `{intent}`")
        st.markdown(f"**Chatbot:** {response}")

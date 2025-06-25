import streamlit as st
import torch
import json
import pandas as pd
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
from nltk.corpus import stopwords
import os

# Page config
st.set_page_config(
    page_title="TanyaRasa - Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        text-align: right;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    .sidebar-content {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except:
        return False

@st.cache_resource
def load_and_preprocess_data():
    """Load and preprocess the intents data"""
    try:
        # Load intents.json
        with open("intents.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Debug: tampilkan info loading
        st.write(f"📊 Loaded {len(data['intents'])} intents from JSON")
        
        # Download NLTK data
        download_nltk_data()
        
        # Initialize preprocessing tools
        try:
            stop_words = set(stopwords.words('indonesian'))
        except:
            stop_words = set()
        
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        # Flatten data
        rows = []
        for intent in data["intents"]:
            tag = intent["tag"]
            patterns = intent["patterns"]
            responses = intent["responses"]
            for pattern in patterns:
                for response in responses:
                    rows.append({
                        "intent": tag,
                        "pattern": pattern,
                        "response": response
                    })
        
        df = pd.DataFrame(rows)
        
        # Debug: tampilkan info dataset
        st.write(f"📋 Created dataset with {len(df)} rows")
        unique_intents = df['intent'].unique()
        st.write(f"📌 Unique intents: {list(unique_intents)}")
        
        # Text preprocessing function
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            words = text.split()
            words = [w for w in words if w not in stop_words]
            text = ' '.join(words)
            stemmed = stemmer.stem(text)
            return stemmed.strip()
        
        # Apply preprocessing
        df["pattern_clean"] = df["pattern"].apply(preprocess_text)
        df["response_clean"] = df["response"].apply(preprocess_text)
        
        return df, True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, False

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        # ✅ PERBAIKAN: Definisikan base model yang benar
        BASE_MODEL_ID = "indolem/indobert-base-uncased"  # Atau model lain yang sesuai
        FINE_TUNED_MODEL_PATH = "athayary/indobert"
        
        # Try to load fine-tuned model first, fallback to base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
            model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL_PATH)
            st.success("✅ Loaded fine-tuned model!")
        except Exception as e:
            st.warning(f"⚠️ Fine-tuned model not found: {str(e)}")
            st.info(f"🔄 Loading base model: {BASE_MODEL_ID}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
            st.warning("⚠️ Using base model (fine-tuned model not found)")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return model, tokenizer, device, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, False

def predict_intent(text, model, tokenizer, device, df):
    """Predict intent and generate response"""
    try:
        # ✅ PERBAIKAN: Tambahkan debug info di awal
        if st.session_state.debug_mode:
            st.write(f"🔍 Original input: '{text}'")
        
        # Preprocess input
        cleaned = text.lower().strip()
        input_text = "Tentukan intent dari kalimat berikut: " + cleaned
        
        if st.session_state.debug_mode:
            st.write(f"🔍 Model input: '{input_text}'")
        
        # Tokenize
        input_enc = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_enc.input_ids,
                attention_mask=input_enc.attention_mask,
                max_length=16,
                num_beams=2,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode prediction
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        intent = preds[0].strip()
        
        # ✅ PERBAIKAN: Tambahkan debug info setelah prediksi
        if st.session_state.debug_mode:
            st.write(f"🔍 Raw prediction: '{preds[0]}'")
            st.write(f"🔍 Processed intent: '{intent}'")
            available_intents = df['intent'].unique()
            st.write(f"🔍 Available intents: {list(available_intents)}")
            st.write(f"🔍 Intent found in dataset: {intent in available_intents}")
        
        # Get response based on intent
        responses = df[df['intent'] == intent]['response'].values
        if len(responses) > 0:
            response = np.random.choice(responses)
        else:
            # ✅ PERBAIKAN: Coba fallback dengan similarity matching
            intent, response = try_similarity_matching(text, df)
            if intent is None:
                # Ultimate fallback
                fallback_responses = [
                    "Maaf, saya tidak mengerti maksud Anda. Bisakah Anda menjelaskan lebih detail?",
                    "Saya sedang belajar untuk memahami Anda lebih baik. Bisa coba dengan kata-kata yang berbeda?",
                    "Hmm, saya belum yakin bagaimana membantu dengan hal ini. Bisa Anda berikan lebih banyak konteks?"
                ]
                response = np.random.choice(fallback_responses)
                intent = "unknown"
        
        return intent, response
    except Exception as e:
        return "error", f"Maaf, terjadi kesalahan: {str(e)}"

def try_similarity_matching(text, df):
    """Try to match using simple similarity"""
    from difflib import SequenceMatcher
    
    text_lower = text.lower().strip()
    best_match = None
    best_score = 0
    
    # Check similarity with all patterns
    for _, row in df.iterrows():
        pattern = row['pattern'].lower()
        score = SequenceMatcher(None, text_lower, pattern).ratio()
        if score > best_score:
            best_score = score
            best_match = row
    
    # If similarity is good enough, use it
    if best_match is not None and best_score > 0.5:  # 50% similarity threshold
        return best_match['intent'], best_match['response']
    
    # Check for common greeting patterns
    greeting_patterns = ['hai', 'halo', 'hello', 'selamat', 'hi']
    if any(pattern in text_lower for pattern in greeting_patterns):
        greeting_responses = df[df['intent'].str.contains('greeting|salam', case=False, na=False)]
        if len(greeting_responses) > 0:
            response = np.random.choice(greeting_responses['response'].values)
            return 'greeting', response
    
    return None, None

def main():
    # Header
    st.markdown('<div class="main-header">🧠 TanyaRasa - Mental Health Chatbot</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("ℹ️ Informasi")
        st.write("""
        **TanyaRasa** adalah chatbot kesehatan mental yang dapat membantu Anda dengan:
        - Konseling dan dukungan emosional
        - Informasi tentang kesehatan mental
        - Saran untuk mengatasi stres dan kecemasan
        - Dukungan motivasi
        """)
        
        st.header("🚀 Status Model")
        if st.session_state.model_loaded:
            st.success("✅ Model siap digunakan")
        else:
            st.warning("⏳ Memuat model...")
        
        # ✅ PERBAIKAN: Tambahkan toggle debug mode
        st.header("🔧 Debug Mode")
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
        
        # ✅ PERBAIKAN: Tambahkan tombol untuk melihat dataset
        if st.button("🔍 Show Dataset Info"):
            if st.session_state.df is not None:
                st.write("📊 Dataset Info:")
                st.write(f"Total rows: {len(st.session_state.df)}")
                intents = st.session_state.df['intent'].unique()
                st.write(f"Unique intents ({len(intents)}): {list(intents)}")
                
                # Show sample patterns for each intent
                for intent in intents[:3]:  # Show first 3
                    patterns = st.session_state.df[st.session_state.df['intent'] == intent]['pattern'].head(2)
                    st.write(f"**{intent}**: {list(patterns)}")
        
        st.header("📊 Statistik Chat")
        st.write(f"Total pesan: {len(st.session_state.chat_history)}")
        
        if st.button("🗑️ Hapus Riwayat Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load model and data if not loaded
        if not st.session_state.model_loaded:
            with st.spinner("Memuat model dan data..."):
                # Load data
                df, data_success = load_and_preprocess_data()
                if data_success:
                    st.session_state.df = df
                
                # Load model
                model, tokenizer, device, model_success = load_model()
                if model_success:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                    st.rerun()
        
        # Chat interface
        if st.session_state.model_loaded:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                
                if not st.session_state.chat_history:
                    st.markdown(
                        '<div class="bot-message">👋 Halo! Saya TanyaRasa, chatbot kesehatan mental. '
                        'Bagaimana saya bisa membantu Anda hari ini?</div>',
                        unsafe_allow_html=True
                    )
                
                for chat in st.session_state.chat_history:
                    st.markdown(
                        f'<div class="user-message">👤 {chat["user"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="bot-message">🤖 [{chat["intent"]}] {chat["bot"]}</div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Input form
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Ketik pesan Anda:",
                    placeholder="Contoh: Saya merasa sedih hari ini...",
                    key="user_input"
                )
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    submit_button = st.form_submit_button("💬 Kirim", type="primary")
                
                if submit_button and user_input.strip():
                    with st.spinner("Memproses..."):
                        intent, response = predict_intent(
                            user_input,
                            st.session_state.model,
                            st.session_state.tokenizer,
                            st.session_state.device,
                            st.session_state.df
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "user": user_input,
                            "bot": response,
                            "intent": intent
                        })
                        
                        st.rerun()
        else:
            st.error("❌ Gagal memuat model. Pastikan file model dan data tersedia.")
    
    with col2:
        st.header("📈 Analytics")
        
        if st.session_state.chat_history:
            # Intent distribution
            intents = [chat["intent"] for chat in st.session_state.chat_history]
            intent_counts = pd.Series(intents).value_counts()
            
            st.subheader("Distribusi Intent")
            st.bar_chart(intent_counts)
            
            # Recent intents
            st.subheader("Intent Terakhir")
            recent_intents = intents[-5:][::-1]  # Last 5, reversed
            for i, intent in enumerate(recent_intents, 1):
                st.write(f"{i}. {intent}")
        else:
            st.info("Mulai chat untuk melihat analytics")

# ✅ PERBAIKAN: Syntax error diperbaiki
if __name__ == "__main__":
    main()
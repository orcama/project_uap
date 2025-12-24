import os

# --- KONFIGURASI PENTING UNTUK TRANSFORMERS ---
# Wajib diset sebelum import tensorflow/transformers agar support model .h5 lama
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# KONFIGURASI PATH
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "DistilBERT": os.path.join(BASE_DIR, "DistilBERTModel"),
    "RoBERTa": os.path.join(BASE_DIR, "RoBERTaModel"),
    "LSTM Hybrid": os.path.join(BASE_DIR, "lstmhybridModel")
}

LSTM_TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")

# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Deteksi Emosi", page_icon="🎭")
LABELS = ['neutral', 'worry', 'happiness', 'sadness']

# -----------------------------------------------------------------------------
# FUNGSI LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_transformer_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"Folder tidak ditemukan: {model_path}")
            return None, None
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error load Transformer: {e}")
        return None, None

@st.cache_resource
def load_lstm_model(model_path):
    try:
        # 1. Pengecekan Path
        if not os.path.exists(model_path):
            st.error(f"Path tidak ditemukan: {model_path}")
            return None

        if model_path.endswith('.h5'):
            full_path = model_path
        else:
            files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
            if not files: 
                st.error(f"Tidak ada file .h5 di {model_path}")
                return None
            full_path = os.path.join(model_path, files[0])

        # 2. DEFINISI CUSTOM OBJECTS (PATCHING KERAS 3 -> KERAS 2)
        # Patch A: InputLayer (Mengatasi error 'batch_shape')
        from tensorflow.keras.layers import InputLayer
        class PatchedInputLayer(InputLayer):
            def __init__(self, batch_shape=None, dtype=None, **kwargs):
                # Keras 3 mungkin mengirim batch_shape & dtype, kita tangani sesuai Keras 2
                if batch_shape is not None:
                    kwargs['batch_input_shape'] = batch_shape
                # Abaikan dtype policy kompleks, ambil float32 default
                if dtype is not None:
                    kwargs['dtype'] = 'float32' 
                super().__init__(**kwargs)

        # Patch B: DTypePolicy (Mengatasi error 'Unknown dtype policy')
        # Kita buat class palsu yang mewarisi Policy dari Keras 2
        from tensorflow.keras.mixed_precision import Policy
        class DTypePolicy(Policy):
            def __init__(self, name="float32", **kwargs):
                # Keras 3 mungkin mengirim dict atau string, kita paksa float32 agar aman
                super().__init__(name="float32")
            
            @classmethod
            def from_config(cls, config):
                # Menerima config dari file .h5 dan mengembalikan instance baru
                return cls(**config)
            
            def get_config(self):
                return {"name": self.name}

        # 3. LOAD MODEL DENGAN PATCH & TANPA COMPILE
        # compile=False sangat PENTING karena Optimizer Keras 3 sering gagal diload di Keras 2
        # Kita hanya butuh model untuk prediksi (inference), jadi tidak butuh optimizer.
        model = tf.keras.models.load_model(
            full_path, 
            custom_objects={
                'InputLayer': PatchedInputLayer,
                'DTypePolicy': DTypePolicy
            },
            compile=False 
        )
        
        return model

    except Exception as e:
        st.error(f"Error load LSTM: {e}")
        # Tampilkan detail error di terminal juga untuk debugging
        print(f"DETAIL ERROR LSTM: {e}")
        return None

@st.cache_resource
def load_lstm_tokenizer():
    try:
        if not os.path.exists(LSTM_TOKENIZER_PATH):
            st.warning("Tokenizer pickle tidak ditemukan.")
            return None
        with open(LSTM_TOKENIZER_PATH, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Gagal load tokenizer: {e}")
        return None

# -----------------------------------------------------------------------------
# UI DASHBOARD
# -----------------------------------------------------------------------------
st.title("🎭 Klasifikasi Emosi")
st.write("Menggunakan mode: **Legacy Keras (tf-keras)**")

selected_model_name = st.sidebar.selectbox("Pilih Model:", list(MODEL_PATHS.keys()))
user_input = st.text_area("Masukkan teks:", height=100)

if st.button("Analisis"):
    if user_input.strip():
        model_path = MODEL_PATHS[selected_model_name]
        probs = None
        
        # --- MODEL TRANSFORMER ---
        if selected_model_name in ["DistilBERT", "RoBERTa"]:
            tokenizer, model = load_transformer_model(model_path)
            if model and tokenizer:
                inputs = tokenizer(user_input, return_tensors="tf", padding=True, truncation=True, max_length=128)
                outputs = model(inputs)
                probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
        
        # --- MODEL LSTM ---
        # ... (Bagian LSTM Hybrid) ...
        elif selected_model_name == "LSTM Hybrid":
            model = load_lstm_model(model_path)
            tok = load_lstm_tokenizer()
            if model and tok:
                seq = tok.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=60) 
                probs = model.predict(padded)[0]

        # --- HASIL ---
        if probs is not None:
            idx = np.argmax(probs)
            st.success(f"Emosi: **{LABELS[idx].upper()}** ({probs[idx]:.2%})")
            st.progress(float(probs[idx]))
            
            with st.expander("Detail Probabilitas"):
                for l, p in zip(LABELS, probs):
                    st.write(f"{l}: {p:.4f}")
    else:
        st.warning("Masukkan teks terlebih dahulu.")
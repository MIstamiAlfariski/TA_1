import random
import json
import pickle
import numpy as np
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. INISIALISASI (LOAD OTAK AI) ---
print("[INFO] Loading Keras Model & Artifacts...")

# Load Model Bi-LSTM
model = load_model('/home/riss/TA_1/model/chat_model.h5')

# Load Tokenizer & Encoder
with open('/home/riss/TA_1/tokenizer/tokenizers.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('/home/riss/TA_1/tokenizer/le.pkl', 'rb') as f:
    le = pickle.load(f)

# Load Database Jawaban (JSON)
# Pastikan path ini sesuai dengan lokasi file json Anda
with open('/home/riss/TA_1/dataset/datahseet.json', 'r') as f: # Sesuaikan nama file json Anda
    intents = json.load(f)

# Simpan respon dalam dictionary agar pencarian cepat
responses = {}
for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']

# Dapatkan Input Shape otomatis dari model
input_shape = model.input_shape[1]
print("[INFO] System Ready!")


# --- 2. FUNGSI UTAMA (Dipanggil oleh app.py) ---
def get_response(msg):
    """
    Fungsi ini menerima teks string dari user, 
    dan mengembalikan teks string jawaban bot.
    """
    try:
        # A. Preprocessing (Harus SAMA PERSIS dengan saat Training)
        # 1. Lowercase & Hapus Punctuation
        msg_processed = [letters.lower() for letters in msg if letters not in string.punctuation]
        msg_processed = ''.join(msg_processed)
        
        # 2. Tokenizing (Ubah ke Angka)
        # texts_to_sequences mengharapkan list of text, jadi bungkus [msg]
        seq = tokenizer.texts_to_sequences([msg_processed])
        
        # 3. Padding (Samakan Panjang)
        padded = pad_sequences(seq, maxlen=input_shape)

        # B. Prediksi
        prediction = model.predict(padded, verbose=0) # verbose=0 agar tidak nyampah di terminal
        prediction_idx = prediction.argmax()
        
        # C. Ambil Tag Hasil Prediksi
        tag = le.inverse_transform([prediction_idx])[0]
        
        # D. Ambil Jawaban Acak
        if tag in responses:
            return random.choice(responses[tag])
        else:
            return "Maaf, saya belum mengerti pertanyaan tersebut."
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return "Terjadi kesalahan pada sistem."

if __name__ == "__main__":
    print("Cek Bot: ", get_response("Halo"))
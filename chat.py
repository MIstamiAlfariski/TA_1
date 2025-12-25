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
    try:
        # A. Preprocessing
        msg_processed = [letters.lower() for letters in msg if letters not in string.punctuation]
        msg_processed = ''.join(msg_processed)
        
        seq = tokenizer.texts_to_sequences([msg_processed])
        
        # Cek apakah kata dikenali (PENTING)
        if not seq or not seq[0]:
             print(f"[DEBUG] Input '{msg}' tidak dikenali sama sekali oleh Tokenizer.")
             return "Maaf, saya tidak mengerti kata-kata tersebut."

        padded = pad_sequences(seq, maxlen=input_shape)

        # B. Prediksi Model
        prediction = model.predict(padded, verbose=0)
        
        # --- LOGIKA THRESHOLD ---
        results = prediction[0]
        prediction_idx = np.argmax(results)
        confidence_score = results[prediction_idx]
        
        tag_prediksi = le.inverse_transform([prediction_idx])[0]

        # --- TAMPILKAN DI TERMINAL (Hanya untuk Developer) ---
        print(f"\n[DEBUG SISTEM]")
        print(f"Pesan User     : {msg}")
        print(f"Tebakan Bot    : {tag_prediksi}")
        print(f"Tingkat Yakin  : {confidence_score:.4f} (atau {confidence_score*100:.2f}%)")
        print(f"------------------------------------------------")

        # Set Threshold agak tinggi (Coba 0.7 atau 70%)
        ERROR_THRESHOLD = 0.50 

        if confidence_score > ERROR_THRESHOLD:
            if tag_prediksi in responses:
                return random.choice(responses[tag_prediksi])
        
        # Jika keyakinan di bawah 70%, tolak menjawab
        return "Maaf, pertanyaan tersebut di luar konteks Fara'idh. Silakan tanya seputar waris."

    except Exception as e:
        print(f"[ERROR] {e}")
        return "Terjadi kesalahan sistem."

if __name__ == "__main__":
    print("Cek Bot: ", get_response("Halo"))
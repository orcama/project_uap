
---

```md
# 🧠 Klasifikasi Deteksi Emosi pada Teks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![NLP](https://img.shields.io/badge/NLP-Emotion%20Classification-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)

## 📌 Deskripsi Proyek
Proyek ini berfokus pada **klasifikasi deteksi emosi pada teks** menggunakan pendekatan **Deep Learning dan Transformer-based Models**. Sistem dirancang untuk mengidentifikasi emosi dari teks (tweet) ke dalam empat kelas utama, yaitu:

- **Neutral**
- **Worry**
- **Happiness**
- **Sadness**

Penelitian ini membandingkan performa tiga model berbeda, yaitu **Hybrid LSTM**, **DistilBERT**, dan **RoBERTa**, serta menyediakan **aplikasi website berbasis Streamlit** untuk pengujian secara interaktif.

---

## 📂 Struktur Proyek
```

├── DistilBERTModel/        # Model DistilBERT tersimpan
├── lstmhybridModel/        # Model Hybrid LSTM tersimpan
├── RoBERTaModel/           # Model RoBERTa tersimpan
├── app.py                  # Aplikasi Streamlit
├── tweet_emotions.csv      # Dataset
├── FIXUAP_PembelajaranMesin.ipynb
├── requirements.txt
└── README.md

````

---

## 📊 Dataset dan Preprocessing

### Dataset
Dataset yang digunakan berupa **dataset tweet emosi** dengan berbagai label emosi. Untuk keperluan penelitian, dataset difilter sehingga hanya mencakup empat label emosi berikut:

- `neutral`
- `worry`
- `happiness`
- `sadness`

### Tahapan Preprocessing
Tahapan preprocessing yang dilakukan meliputi:
1. **Seleksi label emosi**
2. **Pembersihan teks**
   - Menghapus URL, mention, hashtag
   - Menghapus karakter khusus dan angka
   - Mengubah teks menjadi huruf kecil
3. **Tokenisasi**
   - Tokenizer Transformer untuk DistilBERT dan RoBERTa
   - Tokenizer Keras untuk Hybrid LSTM
4. **Padding & Truncation**
5. **Split data** (training dan testing)

---

## 🤖 Model yang Digunakan

### 1️⃣ Hybrid LSTM
Model ini menggunakan **Embedding Layer** dan **Bidirectional LSTM** untuk menangkap pola urutan kata dalam teks.

**Karakteristik:**
- Model non-pretrained
- Digunakan sebagai baseline
- Arsitektur sederhana dan ringan

📁 Folder: `lstmhybridModel/`

---

### 2️⃣ DistilBERT
DistilBERT merupakan versi ringan dari BERT dengan performa yang tetap optimal untuk klasifikasi teks.

**Karakteristik:**
- Transformer-based
- Lebih efisien dan cepat
- Akurasi tinggi pada klasifikasi emosi

📁 Folder: `DistilBERTModel/`

---

### 3️⃣ RoBERTa
RoBERTa adalah pengembangan dari BERT dengan optimasi pretraining yang menghasilkan performa NLP yang lebih kuat.

**Karakteristik:**
- Representasi konteks lebih baik
- Performa paling stabil
- Cocok untuk analisis emosi berbasis teks

📁 Folder: `RoBERTaModel/`

---

## 📈 Hasil Evaluasi dan Analisis

Evaluasi model dilakukan menggunakan **akurasi dan classification report**. Perbandingan hasil ketiga model ditunjukkan pada tabel berikut:

| Nama Model   | Akurasi     | Hasil Evaluasi |
|-------------|------------|----------------|
| Hybrid LSTM | Sedang     | Kesulitan menangani konteks kompleks dan kelas minoritas |
| DistilBERT  | Tinggi     | Performa stabil dengan precision dan recall yang baik |
| RoBERTa     | Tertinggi  | F1-score paling konsisten di seluruh kelas emosi |

### Analisis Singkat
- **RoBERTa** memberikan performa terbaik secara keseluruhan
- **DistilBERT** menjadi alternatif efisien dengan akurasi mendekati RoBERTa
- **Hybrid LSTM** berfungsi sebagai baseline namun kurang optimal untuk teks kompleks

---

## 🌐 Panduan Menjalankan Website Secara Lokal

### 1️⃣ Clone Repository
```bash
git clone <repository-url>
cd <nama-folder-project>
````

### 2️⃣ Install Dependensi

```bash
pip install -r requirements.txt
```

### 3️⃣ Jalankan Aplikasi

```bash
streamlit run app.py
```

### 4️⃣ Akses Website

Buka browser dan akses:

```
http://localhost:8501
```

Pengguna dapat memasukkan teks dan memilih model untuk melihat hasil klasifikasi emosi secara langsung.

---

## 🚀 Teknologi yang Digunakan

* Python
* TensorFlow / Keras
* Hugging Face Transformers
* Scikit-learn
* Streamlit

---

## 📌 Catatan

Proyek ini dikembangkan untuk keperluan **akademik dan penelitian** dalam bidang **Natural Language Processing (NLP)** dan **Data Science**.

---

## 👤 Author

**Nizam Avif Anhari**
Mahasiswa Data Science

---

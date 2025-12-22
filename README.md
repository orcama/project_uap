Siap.
Berikut **README.md dalam 1 cell utuh**, **siap copy–paste langsung** tanpa perlu edit apa pun.

````md
# 🧠 Klasifikasi Deteksi Emosi pada Teks

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan **klasifikasi deteksi emosi pada teks** menggunakan pendekatan **Deep Learning dan Transformer-based Models**. Sistem ini mengklasifikasikan emosi dari teks (tweet) ke dalam empat kelas utama, yaitu **neutral**, **worry**, **happiness**, dan **sadness**.

Penelitian ini membandingkan performa tiga model berbeda, yaitu **Hybrid LSTM**, **DistilBERT**, dan **RoBERTa**. Selain itu, proyek ini juga menyediakan **aplikasi website berbasis Streamlit** agar pengguna dapat melakukan pengujian klasifikasi emosi secara interaktif.

---

## 📊 Dataset dan Preprocessing

### Dataset
Dataset yang digunakan adalah **Emotion Detection from Text Dataset** yang tersedia secara publik di Kaggle:

🔗 https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text

Dataset ini berisi data teks berupa tweet yang telah diberi label emosi. Dari keseluruhan label yang tersedia, penelitian ini hanya menggunakan **empat kelas emosi**, yaitu:
- `neutral`
- `worry`
- `happiness`
- `sadness`

Dataset **tidak disertakan langsung** dalam repository untuk menjaga ukuran repositori tetap ringan dan mengikuti praktik GitHub yang baik.

### Preprocessing
Tahapan preprocessing data teks yang dilakukan meliputi:
1. Seleksi label emosi (hanya 4 kelas)
2. Pembersihan teks (menghapus URL, mention, hashtag, karakter khusus, dan angka)
3. Mengubah teks menjadi huruf kecil (lowercase)
4. Tokenisasi  
   - Tokenizer Transformer untuk DistilBERT dan RoBERTa  
   - Tokenizer Keras untuk Hybrid LSTM
5. Padding dan truncation untuk menyeragamkan panjang input
6. Pembagian data menjadi data latih dan data uji

---

## 🤖 Model yang Digunakan

### 1️⃣ Hybrid LSTM
Model Hybrid LSTM menggunakan **Embedding Layer** dan **Bidirectional LSTM** untuk mempelajari pola urutan kata dalam teks. Model ini dilatih dari awal tanpa pretrained model dan digunakan sebagai **baseline** pembanding.

📁 Folder model: `lstmhybridModel/`

---

### 2️⃣ DistilBERT
DistilBERT merupakan versi ringan dari BERT yang tetap mempertahankan performa tinggi dengan jumlah parameter yang lebih sedikit. Model ini di-fine-tune untuk tugas klasifikasi emosi pada teks.

📁 Folder model: `DistilBERTModel/`

---

### 3️⃣ RoBERTa
RoBERTa adalah pengembangan dari BERT dengan optimasi pada proses pretraining sehingga mampu menghasilkan representasi bahasa yang lebih kuat dan stabil untuk klasifikasi teks.

📁 Folder model: `RoBERTaModel/`

---

## 📈 Hasil Evaluasi dan Analisis Perbandingan

Evaluasi model dilakukan menggunakan **akurasi dan classification report**. Tabel berikut menunjukkan perbandingan performa ketiga model:

| Nama Model   | Akurasi    | Hasil Evaluasi |
|-------------|------------|----------------|
| Hybrid LSTM | Sedang     | Performa lebih rendah, terutama pada teks dengan konteks kompleks |
| DistilBERT  | Tinggi     | Performa stabil dengan keseimbangan precision dan recall yang baik |
| RoBERTa     | Tertinggi  | Memberikan performa terbaik dengan f1-score paling konsisten |

### Analisis
Model berbasis Transformer (**RoBERTa dan DistilBERT**) menunjukkan performa yang lebih baik dibandingkan Hybrid LSTM. **RoBERTa** memberikan hasil terbaik secara keseluruhan, sedangkan **DistilBERT** menjadi alternatif yang lebih efisien dengan performa mendekati RoBERTa.

---


## 🌐 Panduan Menjalankan Sistem Website Secara Lokal

### 1️⃣ Clone Repository
```bash
git clone <repository-url>
cd <nama-folder-project>
````

### 2️⃣ Install Dependensi

```bash
pip install -r requirements.txt
```

### 3️⃣ Pastikan Model Tersedia

Pastikan folder berikut sudah tersedia dan berisi model hasil training:

```
DistilBERTModel/
lstmhybridModel/
RoBERTaModel/
```

### 4️⃣ Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

### 5️⃣ Akses Website

Buka browser dan akses:

```
http://localhost:8501
```

Pengguna dapat memasukkan teks dan memilih model untuk melihat hasil klasifikasi emosi secara langsung.

---

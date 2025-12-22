# Klasifikasi Deteksi Emosi pada Teks

Ini adalah proyek untuk mendeteksi emosi dari teks menggunakan model machine learning berbasis deep learning. Proyek ini mencakup training model menggunakan dataset tweet emotions, serta deployment sederhana melalui aplikasi web berbasis Streamlit.

## Deskripsi Proyek
Proyek ini bertujuan untuk mengklasifikasikan emosi dari teks input pengguna, seperti tweet atau kalimat pendek. Emosi yang dideteksi meliputi: neutral, worry, happiness, sadness (berdasarkan label dataset). Proyek ini melibatkan:
- Preprocessing data teks.
- Training tiga model berbeda: DistilBERT (Transformer-based), RoBERTa (Transformer-based), dan LSTM Hybrid (RNN-based).
- Deployment melalui dashboard web Streamlit untuk analisis emosi secara interaktif.
- Model disimpan di folder: `DistilBERTModel`, `lstmhybridModel`, dan `RoBERTaModel`.

Proyek ini dibangun menggunakan Python, TensorFlow, Hugging Face Transformers, dan Streamlit. Cocok untuk pemula di bidang NLP dan sentiment analysis.

## Dataset dan Preprocessing
Dataset yang digunakan adalah **Emotion Detection from Text** dari Kaggle: [Link Dataset](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text).

- **Deskripsi Dataset**: Dataset berisi sekitar 40.000 tweet dengan label emosi seperti neutral, worry, happiness, sadness, dll. Kolom utama: `tweet_id`, `sentiment` (label emosi), dan `content` (teks tweet).
- **Preprocessing**:
  - **Cleaning Teks**: Menghapus URL, mention (@username), hashtag, angka, dan karakter khusus. Lowercasing teks dan stemming/lemmatization menggunakan library seperti NLTK atau langsung via tokenizer model.
  - **Tokenization**: Untuk model Transformer (DistilBERT dan RoBERTa), menggunakan AutoTokenizer dari Hugging Face dengan max_length=128, padding, dan truncation. Untuk LSTM Hybrid, menggunakan Tokenizer dari Keras dengan sequences padding ke max_length=100.
  - **Splitting Data**: Dataset dibagi menjadi train (80%) dan test (20%) menggunakan train_test_split dari scikit-learn.
  - **Handling Imbalance**: Menggunakan class_weights untuk menangani ketidakseimbangan label (misalnya, emosi 'neutral' lebih dominan).
  - **Augmentasi (Opsional)**: Tidak diterapkan secara eksplisit, tapi bisa ditambahkan untuk meningkatkan variasi data.

Contoh data dari dataset:
```
tweet_id,sentiment,content
1956967341,empty,@tiffanylue i know  i was listenin to bad habit earlier and i started freakin at his part =[
1956967666,sadness,Layin n bed with a headache  ughhhh...waitin on your call...
```

## Penjelasan Ketiga Model yang Digunakan
1. **DistilBERT**:
   - Model Transformer berbasis BERT yang lebih ringan (distilled version dari BERT-base).
   - Digunakan TFAutoModelForSequenceClassification dari Hugging Face.
   - Training: Freeze base layers, fine-tune classifier dengan Adam optimizer (lr=1e-5), SparseCategoricalCrossentropy loss.
   - Keunggulan: Cepat dan efisien untuk deployment, akurasi tinggi pada tugas NLP.

2. **RoBERTa**:
   - Model Transformer yang dioptimasi dari BERT (Robustly Optimized BERT Pretraining Approach).
   - Digunakan TFAutoModelForSequenceClassification dari Hugging Face (pretrained 'roberta-base').
   - Training: Mirip DistilBERT, freeze base layers, fine-tune dengan Adam (lr=1e-4), menggunakan class_weights untuk imbalance.
   - Keunggulan: Lebih baik pada pemahaman konteks dinamis, dilatih lebih lama pada data teks besar.

3. **LSTM Hybrid**:
   - Model berbasis RNN (Recurrent Neural Network) dengan layer LSTM bidirectional + embedding.
   - Arsitektur: Embedding layer, LSTM (hybrid mungkin termasuk dense layers atau GloVe embeddings).
   - Training: Load dari file .h5, menggunakan tokenizer pickle untuk sequences, predict dengan pad_sequences (maxlen=100).
   - Keunggulan: Lebih sederhana dan cepat untuk training pada dataset kecil, tapi kurang powerful dibanding Transformer untuk konteks panjang.

Model-model ini dilatih pada notebook `FIXUAP_PembelajaranMesin.ipynb` dan disimpan untuk inference di app Streamlit.

## Hasil Evaluasi dan Analisis Perbandingan
Berdasarkan evaluasi pada data test (dari notebook training):
- Metrik: Accuracy, Precision, Recall, F1-Score (macro average untuk multi-class).
- Analisis: Transformer (DistilBERT & RoBERTa) unggul karena pre-trained pada data besar, sementara LSTM lebih cepat tapi kurang akurat pada emosi ambigu.

| Nama Model    | Akurasi | Hasil Evaluasi (Precision/Recall/F1-Score Macro Avg) |
|---------------|---------|------------------------------------------------------|
| LSTM Hybrid  | 0.42   | Precision: 0.43, Recall: 0.44, F1: 0.42 - Unggul pada emosi sadness (recall 0.60), tapi rendah pada worry (recall 0.21). |
| DistilBERT   | 0.48   | Precision: 0.48, Recall: 0.48, F1: 0.47 - Lebih seimbang, unggul pada happiness (f1 0.54), tapi rendah pada sadness (f1 0.38). |
| RoBERTa      | 0.47   | Precision: 0.47, Recall: 0.50, F1: 0.47 - Unggul pada happiness (recall 0.67), tapi rendah pada worry (recall 0.30). |

Catatan: Nilai di atas didasarkan pada run terakhir di notebook (mungkin variatif tergantung seed). DistilBERT paling akurat, tapi lebih berat computationally. LSTM cocok untuk perangkat low-end.

## Panduan Menjalankan Sistem Website Secara Lokal
1. **Persyaratan**:
   - Python 3.8+.
   - Install dependensi: Jalankan `pip install -r requirements.txt` (buat file ini jika belum ada, isi: streamlit, numpy, tensorflow, transformers, pickle5, tensorflow.keras).

2. **Clone Repository**:
   ```
   git clone <repo-url>
   cd <repo-folder>
   ```

3. **Siapkan Model**:
   - Pastikan folder model (`DistilBERTModel`, `lstmhybridModel`, `RoBERTaModel`) dan file `tokenizer.pickle` ada di root directory.
   - Jika tidak, train ulang via notebook `FIXUAP_PembelajaranMesin.ipynb`.

4. **Jalankan Aplikasi**:
   ```
   streamlit run app.py
   ```
   - Buka browser di `http://localhost:8501`.
   - Pilih model dari sidebar, masukkan teks, klik "Analisis" untuk deteksi emosi.
   - Output: Label emosi, probabilitas, dan progress bar.

5. **Troubleshooting**:
   - Jika error legacy Keras: Set `os.environ["TF_USE_LEGACY_KERAS"] = "1"` di app.py.
   - Pastikan TensorFlow kompatibel (versi 2.x).
   - Untuk LSTM: Sesuaikan maxlen=100 jika error padding.

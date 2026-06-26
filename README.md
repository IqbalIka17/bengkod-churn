# 📊 Prediksi Churn Pelanggan — UAS Bengkel Koding Data Science

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-bengkod.streamlit.app/)

---

## 👤 Identitas Mahasiswa

| | |
|---|---|
| **Nama** | Iqbal Ika Rahmawan |
| **NIM** | A11.2023.15482 |
| **Matakuliah** | Bengkel Koding |
| **Kelompok** | Data Science 04 |

---

## 📌 Deskripsi Proyek

Proyek ini merupakan tugas **Ujian Akhir Semester (UAS)** Bengkel Koding Data Science yang bertujuan membangun model prediksi **churn pelanggan** menggunakan dataset *Sales and Marketing Customer*.

**Churn pelanggan** adalah kondisi ketika seorang pelanggan berhenti menggunakan layanan atau tidak lagi melakukan aktivitas pembelian. Model yang dibangun memprediksi apakah seorang pelanggan akan churn (`1`) atau bertahan (`0`).

---

## 📂 Struktur Repository

```
bengkod-churn/
├── app.py                      # Aplikasi Streamlit
├── requirements.txt            # Dependensi Python
├── best_model_rf.joblib        # Model Random Forest terbaik
├── scaler.joblib               # StandardScaler
├── selected_features.joblib    # Daftar fitur terpilih
├── datasets/                   # Dataset Sales & Marketing Customer
├── notebook/                   # Jupyter Notebook pengerjaan UAS
└── README.md                   # Dokumentasi proyek
```

---

## 🗂️ Dataset

- **Sumber:** [Sales and Marketing Dataset — Kaggle](https://www.kaggle.com/datasets/bhaskerpaul/sales-and-marketing-dataset)
- **Jumlah sampel:** 15.000 records
- **Jumlah fitur:** 30 kolom
- **Target:** `churn` (0 = bertahan, 1 = churn)

---

## 🔬 Tahapan Pengerjaan

### 1. Exploratory Data Analysis (EDA)
- Menampilkan informasi dataset dan statistik deskriptif
- Visualisasi missing value
- Distribusi variabel target (churn)
- Heatmap korelasi antar fitur numerik

### 2. Direct Modeling
Melatih tiga model tanpa preprocessing:

| Model | Kategori |
|-------|----------|
| Logistic Regression | Konvensional |
| Random Forest | Ensemble Bagging |
| Voting Classifier (LR + LinearSVC + KNN) | Ensemble Voting |

### 3. Modeling dengan Preprocessing
Pipeline preprocessing yang diterapkan:
- ✅ Penanganan missing value (imputasi median/modus)
- ✅ Penghapusan duplikat
- ✅ Feature engineering dari kolom tanggal (`tenure_days`, `days_since_last_purchase`)
- ✅ Capping outlier (IQR 1.5×)
- ✅ Encoding kategorikal (Label Encoding + One-Hot Encoding)
- ✅ StandardScaler (setelah train-test split)
- ✅ SMOTE untuk menangani imbalanced data (85% vs 15%)

### 4. Hyperparameter Tuning & Feature Selection
- Analisis feature importance dengan Random Forest
- Seleksi fitur berdasarkan kumulatif importance ≥ 95%
- GridSearchCV untuk Logistic Regression dan Voting Classifier
- RandomizedSearchCV untuk Random Forest (n_iter=10, cv=3)

---

## 📈 Hasil Evaluasi Model

### Perbandingan F1-Score Antar Skenario

| Model | Direct | Preprocessing | Tuning |
|-------|--------|---------------|--------|
| Logistic Regression | 0.2118 | 0.4777 | 0.4691 |
| Random Forest | 0.4255 | 0.5686 | **0.6252** |
| Voting Classifier | 0.0532 | 0.4987 | 0.4699 |

### 🏆 Model Terbaik: Random Forest (Hyperparameter Tuning)

| Metrik | Nilai |
|--------|-------|
| Accuracy | 0.8573 |
| Precision | 0.5235 |
| Recall | 0.7761 |
| **F1-Score** | **0.6252** |

---

## 🚀 Deployment

Aplikasi dideploy menggunakan **Streamlit Cloud** dan dapat diakses secara publik:

🔗 **[churn-bengkod.streamlit.app](https://churn-bengkod.streamlit.app/)**

### Fitur Aplikasi
- 📝 Form input data pelanggan (20 fitur)
- 🔍 Prediksi churn secara real-time
- 📊 Visualisasi probabilitas prediksi
- 💡 Rekomendasi tindakan retensi
- 🔎 Detail fitur yang digunakan model
- 📈 Tab Feature Importance interaktif

---

## 🛠️ Teknologi yang Digunakan

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)

| Library | Kegunaan |
|---------|----------|
| `pandas` | Manipulasi data |
| `numpy` | Komputasi numerik |
| `scikit-learn` | Machine learning & preprocessing |
| `imbalanced-learn` | SMOTE untuk imbalanced data |
| `matplotlib` | Visualisasi |
| `joblib` | Simpan & load model |
| `streamlit` | Web app deployment |

---

## ⚙️ Cara Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/Iqballka17/bengkod-churn.git
cd bengkod-churn

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

---

*UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro*
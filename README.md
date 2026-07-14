# Aura Gold: Prediksi Emas & ROI Tracker (Vercel Ready) 🪙📈

Aplikasi web Machine Learning end-to-end yang dibangun menggunakan arsitektur modern **FastAPI** (sebagai Backend API) dan **Vanilla HTML/JS/CSS** (sebagai Frontend). Proyek ini dirancang agar kompatibel penuh untuk dideploy ke **Vercel (Serverless Functions)** secara gratis.

Aplikasi ini menyelaraskan harga emas dunia (`GC=F`) secara live, mengonversinya ke rupiah per gram secara real-time berdasarkan kurs live USD ke IDR (`USDIDR=X`), melatih model Random Forest Regressor secara *on-the-fly*, dan menampilkan visualisasi interaktif pergerakan harga emas beserta perhitungan ROI investasi.

---

## 🚀 Fitur Utama & Keunggulan Arsitektur

- **Backend FastAPI Serverless:** Endpoint `/api/predict` berjalan sebagai Vercel Serverless Python Function. Sangat cepat, hemat sumber daya, dan tanpa server permanen.
- **Frontend Murni (Vanilla Web):** Tanpa framework berat. HTML5, CSS3 kustom dengan estetika *Glassmorphism* gelap elegan, dan Vanilla JS untuk performa render maksimal.
- **Data Bursa & Kurs Live:** Pengambilan data harga emas dunia dan kurs USD/IDR secara otomatis menggunakan library `yfinance`, dikonversi otomatis ke satuan berat terpopuler di Indonesia: **Rupiah per Gram**.
- **Model Prediksi Machine Learning:** Model Random Forest melatih dirinya sendiri secara dinamis berdasarkan data bursa historis terbaru, memberikan sinyal perdagangan esok hari (**BELI**, **JUAL**, **HOLD**).
- **Grafik Interaktif Plotly.js:** Visualisasi pergerakan emas 1 tahun terakhir lengkap dengan garis Moving Average (MA 7-hari & MA 21-hari).
- **Kalkulator ROI Instan:** Menghitung total investasi awal, nilai saat ini, profit/loss nominal, dan persentase ROI secara instan langsung di sisi klien (*client-side*) menggunakan JavaScript.

---

## 📂 Struktur Direktori Proyek

```text
Physical Gold Investment Tracker & Trend Predictor/
├── api/
│   └── index.py        # Backend FastAPI (Serverless Function)
├── index.html          # Frontend Murni (HTML, CSS kustom, & JavaScript)
├── vercel.json         # Konfigurasi perutean & pembangun Vercel
├── requirements.txt    # Dependensi modul Python (FastAPI, Scikit-Learn, dll.)
├── .gitignore          # File pengecualian pelacakan Git
└── README.md           # Dokumentasi lengkap proyek (File ini)
```

---

## 💻 Panduan Menjalankan Proyek Secara Lokal

Ikuti langkah-langkah di bawah untuk mencoba aplikasi ini di komputer lokal Anda:

### Prasyarat
Pastikan Anda telah menginstal **Python 3.8** atau versi terbaru.

### Langkah 1: Buat Virtual Environment
Buat virtual environment untuk memisahkan instalasi library:
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Langkah 2: Instal Dependensi
Pasang pustaka yang diperlukan:
```bash
pip install -r requirements.txt
```

### Langkah 3: Jalankan Backend FastAPI
Jalankan server API lokal menggunakan Uvicorn:
```bash
python -m uvicorn api.index:app --reload
```
Server backend akan aktif di alamat `http://127.0.0.1:8000`.

### Langkah 4: Jalankan Frontend
Buka file `index.html` secara langsung di browser Anda (klik dua kali pada file) atau gunakan ekstensi editor seperti *Live Server* di VS Code.
- *Catatan:* JavaScript di dalam `index.html` memiliki fitur pendeteksi bursa lokal. Jika dijalankan di browser secara lokal lewat protokol `file://`, ia akan otomatis menembak server FastAPI lokal Anda di port `8000`.

---

## ☁️ Cara Deploy ke Vercel

Aplikasi ini siap dideploy ke Vercel dalam hitungan menit menggunakan konfigurasi [vercel.json](file:///D:/Physical%20Gold%20Investment%20Tracker%20&%20Trend%20Predictor/vercel.json) yang telah disediakan.

### Opsi A: Menggunakan GitHub (Sangat Direkomendasikan)
1. Buat repositori baru di akun GitHub Anda.
2. Push proyek ini ke GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - FastAPI refactor"
   git branch -M main
   git remote add origin <url-repo-github-anda>
   git push -u origin main
   ```
3. Masuk ke dashboard [Vercel](https://vercel.com/) Anda.
4. Klik **Add New** > **Project** lalu impor repositori GitHub tersebut.
5. Vercel akan otomatis mengenali proyek Python dan melakukan *build* serta *deployment* secara instan.

### Opsi B: Menggunakan Vercel CLI
Jika Anda menginstal Vercel CLI secara global, cukup jalankan perintah di direktori proyek:
```bash
vercel
```
Dan ikuti instruksi yang tampil di layar terminal untuk melakukan deployment cepat.

---

## 🛡️ Catatan Hukum (Disclaimer)
Aura Gold dirancang khusus sebagai portofolio machine learning dan visualisasi data keuangan. Pengembang tidak bertanggung jawab atas keputusan investasi finansial apa pun yang dibuat berdasarkan proyek ini.

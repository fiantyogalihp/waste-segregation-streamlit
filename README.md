# Waste Segregation YOLO (Streamlit)

Aplikasi web sederhana untuk inferensi *object detection* berbasis YOLO pada kasus **waste segregation** dengan kelas:

* `glass`
* `paper`
* `plastic_bags`

Aplikasi ini dirancang untuk:

* memenuhi kebutuhan **UAS Computer Vision**
* **deploy gratis** menggunakan **Streamlit Community Cloud**
* menggunakan **model YOLO `.pt` dari GitHub Release (auto-download + cache)**

---

## 1. Struktur Repository (Disarankan)

```
.
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

Catatan:

* **Model `.pt` tidak disimpan di repo**
* Model akan **diunduh otomatis dari GitHub Release Asset**

---

## 2. Cara Menjalankan Aplikasi (Local)

### Prasyarat

* Python **3.10 atau 3.11**
* OS: Linux / macOS / Windows
* GPU **tidak wajib** (Streamlit Cloud pakai CPU)

### Langkah Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Akses di browser:

```
http://localhost:8501
```

---

## 3. Model YOLO (.pt) – GitHub Release + Cache

### Kenapa GitHub Release?

* Ukuran model ± **40MB**
* Repo tetap ringan
* Aman dan stabil untuk Streamlit Cloud

### Cara Menyiapkan Model

1. Buka GitHub repository
2. Buat **Release** (contoh: `v1.0.0`)
3. Upload file model:

   ```
   best.pt
   ```
4. Copy URL asset model

Contoh URL:

```
https://github.com/username/repo-name/releases/download/v1.0.0/best.pt
```

---

## 4. Konfigurasi Environment Variable (Wajib untuk Deploy)

Aplikasi membaca model dari environment variable berikut:

| Variable       | Keterangan                       |
| -------------- | -------------------------------- |
| `MODEL_URL`    | URL GitHub Release Asset (`.pt`) |
| `MODEL_SHA256` | (Opsional) checksum model        |

### Contoh (Streamlit Cloud → Settings → Secrets)

```toml
MODEL_URL = "https://github.com/username/repo/releases/download/v1.0.0/best.pt"
MODEL_SHA256 = "isi_sha256_jika_ada"
```

Model akan di-cache otomatis di:

```
~/.cache/waste-yolo/best.pt
```

---

## 5. Deploy ke Streamlit Community Cloud (Gratis)

### Langkah Deploy

1. Push repo ke GitHub
2. Buka: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Klik **New App**
4. Pilih:

   * Repository
   * Branch (mis. `main`)
   * Main file: `app.py`
5. Tambahkan **Secrets** (MODEL_URL)
6. Klik **Deploy**

Selesai. App akan online dengan URL publik.

---

## 6. Troubleshooting Umum

### Model gagal download / error `.tmp`

Pastikan:

* Direktori cache dibuat dengan:

  ```python
  os.makedirs(cache_dir, exist_ok=True)
  ```
* Model sudah tersedia di GitHub Release
* `MODEL_URL` benar (direct asset, bukan halaman repo)

### App lambat saat pertama run

Normal. Model sedang di-download (sekali saja).

---

## 7. Catatan untuk Laporan / Video UAS

Alur yang disarankan untuk presentasi:

1. Dataset & augmentasi (Roboflow)
2. Training YOLO
3. Evaluasi (loss, mAP, confusion matrix)
4. Deployment Streamlit
5. Live demo upload gambar

---

## 8. Teknologi yang Digunakan

* Python
* Ultralytics YOLO
* Streamlit
* GitHub Releases
* Roboflow (dataset & preprocessing)


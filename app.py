import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Ultralytics YOLO
from ultralytics import YOLO

# -----------------------------
# CONFIG (ubah sesuai repo kamu)
# -----------------------------
APP_NAME = "waste-yolo"
CACHE_DIR = Path.home() / ".cache" / APP_NAME
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Wajib: URL direct-download dari GitHub Release Asset
# Contoh format:
# https://github.com/<user>/<repo>/releases/download/<tag>/best.pt
DEFAULT_MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/fiantyogalihp/waste-segregation-streamlit/releases/download/v1.0.0/best.pt"  # <-- GANTI INI
)

# Nama file model yang disimpan di cache
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best.pt")

# Opsional (disarankan): isi sha256 agar aman dan deterministik
# Cara dapat sha256: `sha256sum best.pt`
MODEL_SHA256 = os.getenv("MODEL_SHA256", "")  # boleh kosong

# Default inference params
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.20"))
DEFAULT_IOU  = float(os.getenv("DEFAULT_IOU", "0.50"))
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "640"))
DEFAULT_MAXDET = int(os.getenv("DEFAULT_MAXDET", "100"))

# -----------------------------
# Utility
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dst: Path, expected_sha256: str = ""):
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    # Bersihkan tmp kalau ada sisa gagal sebelumnya
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    try:
        # Download ke tmp
        urllib.request.urlretrieve(url, tmp.as_posix())

        # Kalau download sukses tapi tmp entah kenapa tidak ada, fallback:
        if not tmp.exists():
            # Jika dst sudah ada & ukuran masuk akal, anggap sudah berhasil
            if dst.exists() and dst.stat().st_size > 1_000_000:
                return True, f"Model sudah ada: {dst}"
            return False, f"Download selesai tapi file tmp tidak ditemukan: {tmp}"

        # Optional SHA256 check
        if expected_sha256:
            got = sha256_file(tmp)
            if got.lower() != expected_sha256.lower():
                tmp.unlink(missing_ok=True)
                return False, f"SHA256 mismatch. expected={expected_sha256} got={got}"

        # Atomic replace
        os.replace(tmp.as_posix(), dst.as_posix())
        return True, f"Model tersimpan di: {dst}"

    except Exception as e:
        # Kalau error tapi dst sudah ada & valid → anggap sukses
        if dst.exists() and dst.stat().st_size > 1_000_000:
            return True, f"Model sudah ada: {dst} (download step error diabaikan: {e})"
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"Gagal download model: {e}"

def ensure_model(model_url: str, filename: str, expected_sha256: str = "") -> Path:
    dst = CACHE_DIR / filename
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Jika sudah ada dan ukurannya masuk akal, pakai langsung
    if dst.exists() and dst.stat().st_size > 1_000_000:
        # Kalau ada sha256, validasi
        if expected_sha256:
            got = sha256_file(dst)
            if got.lower() != expected_sha256.lower():
                st.warning("Model ada tapi SHA256 berbeda. Download ulang...")
                dst.unlink(missing_ok=True)
            else:
                return dst
        else:
            return dst

    ok, msg = download_file(model_url, dst, expected_sha256.strip())
    if not ok:
        st.error(msg)
        st.stop()
    st.success(msg)
    return dst

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    # cache_resource memastikan model tidak reload setiap rerun
    return YOLO(model_path)

def yolo_predict(
    model: YOLO,
    image: Image.Image,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Return:
      - annotated_image (RGB np array)
      - table (pandas DataFrame)
    """
    # Ultralytics menerima PIL / np
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False
    )

    r = results[0]
    annotated = r.plot()  # BGR uint8
    annotated = annotated[:, :, ::-1]  # BGR -> RGB

    rows: List[Dict] = []
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        for b in boxes:
            cls_id = int(b.cls.item()) if b.cls is not None else -1
            conf_v = float(b.conf.item()) if b.conf is not None else 0.0
            xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            name = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") else str(cls_id)
            rows.append({
                "label": name,
                "class_id": cls_id,
                "confidence": round(conf_v, 4),
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    return annotated, df

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="Waste Segregation YOLO",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Waste Segregation - YOLO Detection (Streamlit)")
st.caption("Kelas: glass, paper, plastic_bags (YOLO .pt dari GitHub Release Asset)")

with st.sidebar:
    st.header("Model & Inference Settings")

    model_url = st.text_input("MODEL_URL (GitHub Release Asset)", value=DEFAULT_MODEL_URL)
    model_sha = st.text_input("MODEL_SHA256 (opsional, rekomendasi)", value=MODEL_SHA256)
    st.text(f"Cache dir: {CACHE_DIR}")

    conf = st.slider("Confidence (conf)", 0.01, 0.99, float(DEFAULT_CONF), 0.01)
    iou  = st.slider("IoU", 0.01, 0.99, float(DEFAULT_IOU), 0.01)

    imgsz = st.selectbox("Image size (imgsz)", options=[512, 640, 768, 960, 1024], index=[512,640,768,960,1024].index(DEFAULT_IMGSZ) if DEFAULT_IMGSZ in [512,640,768,960,1024] else 1)
    max_det = st.selectbox("Max detections (max_det)", options=[50, 100, 200, 300], index=[50,100,200,300].index(DEFAULT_MAXDET) if DEFAULT_MAXDET in [50,100,200,300] else 1)

    st.divider()
    st.write("Tips:")
    st.write("- Jika `plastic_bags` sering miss, turunkan `conf` (mis. 0.15–0.25).")
    st.write("- Jika box terlalu banyak/overlap, naikkan `iou` (mis. 0.55–0.70).")

# Pastikan model tersedia + load
model_path = ensure_model(model_url, MODEL_FILENAME, model_sha.strip())
model = load_model(model_path.as_posix())

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png", "webp"])

    use_sample = st.checkbox("Pakai sample image (jika tidak upload)", value=False)

    image: Optional[Image.Image] = None
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
    elif use_sample:
        # Sample sederhana (generate)
        w, h = 640, 480
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, :] = 240
        image = Image.fromarray(arr, "RGB")

    if image is not None:
        st.image(image, caption="Original", use_container_width=True)

with col2:
    st.subheader("Output")
    if image is None:
        st.info("Upload gambar dulu, atau centang sample image.")
    else:
        with st.spinner("Running inference..."):
            annotated, df = yolo_predict(
                model=model,
                image=image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det
            )

        st.image(annotated, caption="Detections", use_container_width=True)

        st.markdown("### Tabel Deteksi")
        if df.empty:
            st.warning("Tidak ada deteksi pada threshold saat ini. Coba turunkan confidence.")
        else:
            st.dataframe(df, use_container_width=True)

        # Ringkasan count per label
        if not df.empty:
            st.markdown("### Ringkasan")
            counts = df["label"].value_counts().rename_axis("label").reset_index(name="count")
            st.dataframe(counts, use_container_width=True)

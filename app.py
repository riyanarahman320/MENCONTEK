import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # <-- PENTING untuk JavaScript
from ultralytics import YOLO
from PIL import Image
import json
import os

# Muat model Anda
# Pastikan 'contek.pt' ada di direktori yang sama
try:
    model = YOLO("contek.pt")
    print("Model 'contek.pt' berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model: {e}")
    model = None

# Buat aplikasi FastAPI
app = FastAPI(title="YOLOv13 Detection API")

# --- Konfigurasi CORS ---
# Ini mengizinkan browser (dari domain manapun) untuk mengakses API Anda.
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua method (GET, POST, dll)
    allow_headers=["*"],  # Izinkan semua header
)
# --- Selesai Konfigurasi CORS ---


@app.get("/")
def read_root():
    return {"message": "Selamat datang di API YOLOv13. Gunakan endpoint /detect."}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Menerima file gambar (dari webcam), menjalankan deteksi, 
    dan mengembalikan hasil dalam format JSON.
    """
    if not model:
        return {"error": "Model tidak berhasil dimuat, periksa log server."}, 500

    contents = await file.read()
    
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Tidak dapat membaca file gambar: {e}"}

    # Jalankan deteksi
    results = model(image, verbose=False) 
    result = results[0]

    # Format output menjadi JSON yang rapi
    output_boxes = []
    for box in result.boxes:
        output_boxes.append({
            "class_id": int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "coordinates": [float(coord) for coord in box.xyxy[0]] # [x1, y1, x2, y2]
        })

    return {
        "filename": file.filename,
        "detections": output_boxes
    }

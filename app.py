import cv2
from ultralytics import YOLO

# Muat model 'best.pt' yang sudah Anda latih
model = YOLO('contek.pt')

# Inisialisasi webcam
# Angka 0 biasanya adalah webcam bawaan (built-in)
# Jika Anda punya banyak kamera, Anda bisa coba ganti ke 1, 2, dst.
cap = cv2.VideoCapture(0)

# Periksa apakah webcam berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Loop untuk membaca frame dari kamera secara terus menerus
while True:
    # Baca satu frame dari webcam
    ret, frame = cap.read()

    # Jika frame tidak berhasil dibaca (misal: kamera dicabut)
    if not ret:
        print("Error: Tidak bisa membaca frame.")
        break

    # Jalankan deteksi YOLO pada frame
    # 'stream=True' disarankan untuk video agar lebih efisien
    results = model(frame, stream=True)

    # Proses hasil deteksi
    for r in results:
        # 'plot()' adalah fungsi bawaan ultralytics untuk menggambar
        # semua kotak pembatas (bounding boxes) dan label pada frame
        annotated_frame = r.plot()

        # Tampilkan frame yang sudah diberi anotasi
        cv2.imshow('Deteksi Mencontek (Tekan q untuk keluar)', annotated_frame)

    # Cek apakah pengguna menekan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Setelah loop selesai, lepaskan webcam
cap.release()
# Tutup semua jendela OpenCV
cv2.destroyAllWindows()
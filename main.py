import pandas as pd
import requests
import face_recognition
import numpy as np
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import os

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("data.xlsx")

known_face_encodings = []
known_face_names = []

# =========================
# SETUP MEDIAPIPE
# =========================
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=5,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# =========================
# FUNGSI HITUNG JARAK
# =========================
def hitung_jarak(p1, p2):
    return int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

# =========================
# FUNGSI DOWNLOAD GAMBAR
# =========================
def download_image(url):
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        session = requests.Session()
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(download_url, stream=True)

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                download_url = f"{download_url}&confirm={value}"
                response = session.get(download_url, stream=True)
                break

        if response.status_code != 200:
            return None
        return response.content
    except Exception as e:
        print(f"[ERROR Download] {e}")
        return None

# =========================
# AUTO-DETECT IVCAM (FIXED)
# =========================
def find_ivcam():
    print("🔍 Mencari iVCam...")

    # ✅ Paksa tunggu iVCam siap
    time.sleep(3)

    kamera_aktif = []
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"[FOUND] Index {i} — resolusi {w}x{h}")
                kamera_aktif.append((i, w, h))
            cap.release()

    if not kamera_aktif:
        print("❌ Tidak ada kamera ditemukan!")
        exit()

    if len(kamera_aktif) == 1:
        idx = kamera_aktif[0][0]
        print(f"✅ Hanya 1 kamera → pakai index {idx}")
        return idx

    # ✅ iVCam biasanya resolusi lebih tinggi dari webcam bawaan
    # Pilih kamera dengan resolusi tertinggi = iVCam
    best = max(kamera_aktif, key=lambda x: x[1] * x[2])
    print(f"✅ Pakai iVCam di index {best[0]} (resolusi {best[1]}x{best[2]})")
    return best[0]

# =========================
# FUNGSI LANDMARK & RASIO
# =========================
def tampilkan_ukuran(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hasil = face_mesh.detect(mp_image)

    if not hasil.face_landmarks:
        return frame

    for face_landmarks in hasil.face_landmarks:
        lm = face_landmarks

        def get_point(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        mata_kiri   = get_point(33)
        mata_kanan  = get_point(263)
        hidung      = get_point(1)
        bibir_atas  = get_point(13)
        bibir_bawah = get_point(14)
        dagu        = get_point(152)
        dahi        = get_point(10)

        jarak_mata         = hitung_jarak(mata_kiri, mata_kanan)
        jarak_hidung_bibir = hitung_jarak(hidung, bibir_atas)
        jarak_bibir        = hitung_jarak(bibir_atas, bibir_bawah)
        jarak_dagu_dahi    = hitung_jarak(dagu, dahi)

        rasio_mata_wajah   = round(jarak_mata / jarak_dagu_dahi, 3) if jarak_dagu_dahi > 0 else 0
        rasio_bibir_mata   = round(jarak_bibir / jarak_mata, 3) if jarak_mata > 0 else 0
        rasio_hidung_bibir = round(jarak_hidung_bibir / jarak_dagu_dahi, 3) if jarak_dagu_dahi > 0 else 0

        for titik in [mata_kiri, mata_kanan, hidung, bibir_atas, bibir_bawah, dagu, dahi]:
            cv2.circle(frame, titik, 4, (0, 255, 255), -1)

        cv2.line(frame, mata_kiri, mata_kanan, (255, 165, 0), 1)
        cv2.line(frame, hidung, bibir_atas, (255, 165, 0), 1)
        cv2.line(frame, bibir_atas, bibir_bawah, (255, 165, 0), 1)
        cv2.line(frame, dagu, dahi, (255, 165, 0), 1)

        y = 30
        infos = [
            f"Mata L-R    : {jarak_mata}px  | rasio: {rasio_mata_wajah}",
            f"Bibir A-B   : {jarak_bibir}px  | rasio: {rasio_bibir_mata}",
            f"Hidung-Bibir: {jarak_hidung_bibir}px  | rasio: {rasio_hidung_bibir}",
            f"Tinggi Wajah: {jarak_dagu_dahi}px (acuan)",
        ]
        for info in infos:
            cv2.putText(frame, info, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            y += 25

    return frame

# =========================
# ENCODING WAJAH + CACHE
# =========================
CACHE_FILE = "face_cache.pkl"

# ✅ Cek apakah cache sudah ada
if os.path.exists(CACHE_FILE):
    print("⚡ Cache ditemukan! Memuat data dari cache...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    known_face_encodings = cache["encodings"]
    known_face_names     = cache["names"]
    print(f"✅ {len(known_face_names)} wajah dimuat dari cache (instan!)")

else:
    # ✅ Belum ada cache → download & encode seperti biasa
    print("📥 Cache belum ada, download & encode foto...")

    for index, row in df.iterrows():
        name = row['Nama Lengkap']
        url  = row['File Wajah']

        try:
            img_bytes = download_image(url)
            if img_bytes is None:
                print(f"[SKIP] {name} - gagal download")
                continue

            nparr = np.frombuffer(img_bytes, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print(f"[SKIP] {name} - gagal decode gambar")
                continue

            h, w = img.shape[:2]
            if w > 1000:
                scale = 1000 / w
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            rgb_img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(
                rgb_img,
                face_recognition.face_locations(rgb_img, model="hog")
            )

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"[OK] {name}")
            else:
                print(f"[SKIP] {name} - wajah tidak terdeteksi di foto")

        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    # ✅ Simpan cache untuk run berikutnya
    if len(known_face_names) > 0:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        print(f"💾 Cache disimpan ke {CACHE_FILE}")

print(f"\nTotal wajah terdaftar: {len(known_face_names)}")

if len(known_face_names) == 0:
    print("❌ Tidak ada wajah yang berhasil di-encode. Cek foto referensi!")
    exit()

# =========================
# BUKA IVCAM
# =========================
ivcam_index = 1 #gantii 1
video_capture = cv2.VideoCapture(ivcam_index, cv2.CAP_DSHOW)
time.sleep(2)

if not video_capture.isOpened():
    print("❌ Kamera tidak bisa dibuka")
    exit()

print("✅ Kamera berhasil dibuka, memulai face recognition...")

process_this_frame = True
face_locations = []
face_names     = []
THRESHOLD      = 0.55
SCALE          = 0.5

# =========================
# FACE RECOGNITION LOOP
# =========================
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Gagal baca frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb_frame   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations     = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings_frame = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings_frame:
            name = "Unknown"
            matches          = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=THRESHOLD)
            face_distances   = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name       = known_face_names[best_match_index]
                confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                name       = f"{name} ({confidence}%)"

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top    = int(top / SCALE)
        right  = int(right / SCALE)
        bottom = int(bottom / SCALE)
        left   = int(left / SCALE)

        color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    frame = tampilkan_ukuran(frame)

    cv2.imshow("Face Recognition - iVCam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
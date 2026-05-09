import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import urllib.request

# ==========================================
# SETTINGS
# ==========================================
INPUT_FOLDER  = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\image_Cropoutput"
FINAL_SIZE    = (400, 400) 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# MODEL DOWNLOADS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

def download_file(url, path, name):
    if not os.path.exists(path):
        print(f"⬇️ Downloading {name}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, path)

# Only downloading the Face Detector now
download_file("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", FACE_MODEL_PATH, "Face Detector")

# Initialize Face Detection Model
detector = cv2.FaceDetectorYN_create(FACE_MODEL_PATH, "", (320, 320), score_threshold=0.6)

print("✅ Systems ready (Standard Resizing Mode).")

# ==========================================
# CORE FUNCTIONS
# ==========================================

def load_and_fix_rotation(path):
    try:
        pil_img = Image.open(path)
        pil_img = ImageOps.exif_transpose(pil_img)
        return pil_img.convert("RGB")
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def detect_face_with_rotation(pil_img):
    for angle in [0, 90, 180, 270]:
        rotated = pil_img.rotate(-angle, expand=True) if angle != 0 else pil_img
        cv_img = cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(cv_img)

        if faces is None:
            continue

        valid_faces = [f for f in faces if f[14] >= 0.7]

        if len(valid_faces) > 1:
            return rotated, "SKIP", angle

        if len(valid_faces) == 1:
            f = valid_faces[0]
            face_box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))

            # Landmark validation
            left_eye   = np.array([f[4],  f[5]])
            right_eye  = np.array([f[6],  f[7]])
            nose       = np.array([f[8],  f[9]])
            mouth_l    = np.array([f[10], f[11]])
            mouth_r    = np.array([f[12], f[13]])

            eye_mid   = (left_eye + right_eye) / 2
            mouth_mid = (mouth_l  + mouth_r)   / 2

            if nose[1] > eye_mid[1] and mouth_mid[1] > eye_mid[1] and right_eye[0] > left_eye[0]:
                if angle != 0:
                    print(f"    🔄 Auto-rotated {angle}°")
                return rotated, face_box, angle

    return pil_img, None, 0

def square_crop_no_stretch(pil_img, face_box):
    w, h = pil_img.size

    if face_box:
        fx, fy, fw, fh = face_box
        cx, cy = fx + fw // 2, fy + fh // 2
        side = int(fh * 3.0)
    else:
        cx, cy = w // 2, h // 2
        side = min(w, h)

    left   = cx - side // 2
    top    = cy - int(side * 0.40)
    right  = left + side
    bottom = top  + side

    # Boundary logic
    if left < 0: left, right = 0, side
    if right > w: right, left = w, max(0, w - side)
    if top < 0: top, bottom = 0, side
    if bottom > h: bottom, top = h, max(0, h - side)

    return pil_img.crop((left, top, right, bottom))

def enhance_and_resize(pil_img):
    """Replaced AI Upscale with High-Quality Lanczos Resizing"""
    # Resize first using Lanczos (best for downscaling/standard scaling)
    pil_res = pil_img.resize(FINAL_SIZE, Image.Resampling.LANCZOS)
    
    # Original enhancement filters
    pil_res = pil_res.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    pil_res = ImageEnhance.Contrast(pil_res).enhance(1.1)
    pil_res = ImageEnhance.Sharpness(pil_res).enhance(1.4)
    
    return pil_res

# ==========================================
# MAIN EXECUTION
# ==========================================
print("\n🚀 Starting 400x400 Processing...")

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    print(f"📸 Processing: {filename}")
    img = load_and_fix_rotation(os.path.join(INPUT_FOLDER, filename))
    if img is None: continue

    img, face_box, angle = detect_face_with_rotation(img)

    if face_box == "SKIP":
        print(f"⏭️  SKIPPED: Multiple faces in {filename}")
        continue

    cropped = square_crop_no_stretch(img, face_box)
    final = enhance_and_resize(cropped)

    out_name = f"{os.path.splitext(filename)[0]}_noupscale.jpg"
    final.save(os.path.join(OUTPUT_FOLDER, out_name), "JPEG", quality=98)
    print(f"✅ Saved 400x400 version")

print(f"\n✨ Done! Files saved in: {OUTPUT_FOLDER}")
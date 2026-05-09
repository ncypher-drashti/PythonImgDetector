import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import urllib.request

# ==========================================
# SETTINGS
# ==========================================
INPUT_FOLDER  = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput_cropped"
FINAL_SIZE = (800, 800)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# FACE DETECTOR LOAD
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet_2023mar.onnx")
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading YuNet model...")
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(url, MODEL_PATH)

detector = cv2.FaceDetectorYN_create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
print("✅ Face detector loaded.")

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

            left_eye  = np.array([f[4],  f[5]])
            right_eye = np.array([f[6],  f[7]])
            nose      = np.array([f[8],  f[9]])
            mouth_l   = np.array([f[10], f[11]])
            mouth_r   = np.array([f[12], f[13]])

            eye_mid   = (left_eye + right_eye) / 2
            mouth_mid = (mouth_l  + mouth_r)   / 2

            eyes_above_nose  = nose[1]      > eye_mid[1]
            eyes_above_mouth = mouth_mid[1] > eye_mid[1]
            eyes_horizontal  = right_eye[0] > left_eye[0]

            if eyes_above_nose and eyes_above_mouth and eyes_horizontal:
                if angle != 0:
                    print(f"   🔄 Auto-rotated {angle}° to fix orientation")
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

    if left < 0:
        left  = 0
        right = side
    if right > w:
        right = w
        left  = max(0, w - side)
    if top < 0:
        top    = 0
        bottom = side
    if bottom > h:
        bottom = h
        top    = max(0, h - side)

    return pil_img.crop((left, top, right, bottom))


def enhance_pixels(pil_img):
    """✅ NO EDSR — INSTANT, SHARP, NO HANGING"""
    pil_img = pil_img.resize(FINAL_SIZE, Image.Resampling.LANCZOS)
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.1)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.4)
    return pil_img


# ==========================================
# MAIN EXECUTION
# ==========================================
print("\n🚀 Starting Proper Processing...")

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    print(f"\n📸 Processing: {filename}")

    # 1. FIX EXIF ROTATION
    pil_img = load_and_fix_rotation(os.path.join(INPUT_FOLDER, filename))
    if pil_img is None:
        continue

    # 2. DETECT FACE + AUTO-ROTATE IF NEEDED
    pil_img, face_box, angle = detect_face_with_rotation(pil_img)

    if face_box == "SKIP":
        print(f"⏭️  SKIPPED {filename}: Multiple faces detected.")
        continue

    # 3. CROP WITHOUT STRETCHING
    cropped_pil = square_crop_no_stretch(pil_img, face_box)

    # 4. ENHANCE + RESIZE — INSTANT, NO AI MODEL
    final_img = enhance_pixels(cropped_pil)

    # 5. SAVE
    out_name = f"{os.path.splitext(filename)[0]}_fixed.jpg"
    final_img.save(os.path.join(OUTPUT_FOLDER, out_name), "JPEG", quality=98)
    print(f"✅ DONE: {filename}" + (f" (rotated {angle}°)" if angle else ""))

print(f"\n✨ Process Finished! Check your folder: {OUTPUT_FOLDER}")
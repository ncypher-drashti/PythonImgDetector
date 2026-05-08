import cv2
import cv2.dnn_superres
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os
import urllib.request

# ==========================================
# SETTINGS
# ==========================================
INPUT_FOLDER  = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput_cropped"
FINAL_SIZE = (400, 400)

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

# ==========================================
# SUPER RESOLUTION MODEL LOAD
# ==========================================
SR_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EDSR_x2.pb")
if not os.path.exists(SR_MODEL_PATH):
    print("⬇️ Downloading EDSR super-resolution model...")
    url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"
    urllib.request.urlretrieve(url, SR_MODEL_PATH)

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(SR_MODEL_PATH)
sr.setModel("edsr", 2)  # 2x upscale
print("✅ Super-resolution model loaded.")

# ==========================================
# CORE FUNCTIONS
# ==========================================

def load_and_fix_rotation(path):
    """SADHI RITE PHOTO LOAD KARE CHE (EXIF ROTATION FIX)"""
    try:
        pil_img = Image.open(path)
        pil_img = ImageOps.exif_transpose(pil_img)
        return pil_img.convert("RGB")
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None


def detect_face_with_rotation(pil_img):
    """
    FACE DETECT KARE CHE — JYARE NAHI MALE, TYARE IMAGE NE ROTATE KARI
    PRAYAS KARE CHE (0°, 90°, 180°, 270°).
    RETURNS: (rotated_pil_img, face_box, angle_used) OR ("SKIP", ...) OR (None, ...)
    """
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

            # LANDMARK-BASED UPRIGHT CHECK
            # YuNet 5 landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
            left_eye  = np.array([f[4],  f[5]])
            right_eye = np.array([f[6],  f[7]])
            nose      = np.array([f[8],  f[9]])
            mouth_l   = np.array([f[10], f[11]])
            mouth_r   = np.array([f[12], f[13]])

            eye_mid   = (left_eye + right_eye) / 2
            mouth_mid = (mouth_l  + mouth_r)   / 2

            # UPRIGHT FACE: nose & mouth must be BELOW eyes (larger y in image coords)
            eyes_above_nose  = nose[1]      > eye_mid[1]
            eyes_above_mouth = mouth_mid[1] > eye_mid[1]

            # EYES HORIZONTAL: left_eye should be LEFT of right_eye
            eyes_horizontal  = right_eye[0] > left_eye[0]

            if eyes_above_nose and eyes_above_mouth and eyes_horizontal:
                if angle != 0:
                    print(f"   🔄 Auto-rotated {angle}° to fix orientation")
                return rotated, face_box, angle

    return pil_img, None, 0


def square_crop_no_stretch(pil_img, face_box):
    """CROP KARE CHE PAN STRETCH THAVA NAHI DE"""
    w, h = pil_img.size
    if face_box:
        fx, fy, fw, fh = face_box
        cx, cy = fx + fw // 2, fy + fh // 2
        side = int(fh * 4.0)
    else:
        cx, cy = w // 2, h // 2
        side = min(w, h)

    left   = max(0, cx - side // 2)
    top    = max(0, cy - int(side * 0.45))
    right  = min(w, left + side)
    bottom = min(h, top + side)

    return pil_img.crop((left, top, right, bottom))


def enhance_pixels(pil_img):
    """SUPER RESOLUTION VADE UPSCALE KARE CHE (NO BLUR)"""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    upscaled = sr.upsample(cv_img)
    return Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))


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

    # 4. SUPER RESOLUTION UPSCALE FIRST (AI-based, no blur)
    enhanced_img = enhance_pixels(cropped_pil)

    # 5. RESIZE TO FINAL SIZE
    final_img = enhanced_img.resize(FINAL_SIZE, Image.Resampling.LANCZOS)

    # 6. SAVE
    out_name = f"{os.path.splitext(filename)[0]}_fixed.jpg"
    final_img.save(os.path.join(OUTPUT_FOLDER, out_name), "JPEG", quality=98)
    print(f"✅ DONE: {filename}" + (f" (rotated {angle}°)" if angle else ""))

print(f"\n✨ Process Finished! Check your folder: {OUTPUT_FOLDER}")
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import urllib.request

# ==========================================
# SETTINGS
# ==========================================
INPUT_FOLDER  = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput_cropped"

FINAL_SIZE = (400, 400)

BLUR_THRESHOLD = 85

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# MODEL PATH
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_detection_yunet_2023mar.onnx"
)

# ==========================================
# DOWNLOAD MODEL
# ==========================================
def download_file(url, path, name):

    if not os.path.exists(path):

        print(f"⬇️ Downloading {name}...")

        opener = urllib.request.build_opener()

        opener.addheaders = [('User-agent', 'Mozilla/5.0')]

        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url, path)

download_file(
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    FACE_MODEL_PATH,
    "Face Detector"
)

# ==========================================
# LOAD FACE DETECTOR
# ==========================================
detector = cv2.FaceDetectorYN_create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    score_threshold=0.6
)

print("✅ System Ready")

# ==========================================
# LOAD IMAGE + FIX ROTATION
# ==========================================
def load_and_fix_rotation(path):

    try:

        pil_img = Image.open(path)

        # Auto-fix mobile photo rotation
        pil_img = ImageOps.exif_transpose(pil_img)

        return pil_img.convert("RGB")

    except Exception as e:

        print(f"❌ Error loading image: {e}")

        return None

# ==========================================
# BLUR CHECK
# ==========================================
def is_blurry(pil_img):

    gray = cv2.cvtColor(
        np.array(pil_img),
        cv2.COLOR_RGB2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    print(f"   🔍 Blur Score: {blur_score:.2f}")

    return blur_score < BLUR_THRESHOLD

# ==========================================
# FACE DETECTION WITH ROTATION
# ==========================================
def detect_face_with_rotation(pil_img):

    for angle in [0, 90, 180, 270]:

        rotated = (
            pil_img.rotate(-angle, expand=True)
            if angle != 0
            else pil_img
        )

        cv_img = cv2.cvtColor(
            np.array(rotated),
            cv2.COLOR_RGB2BGR
        )

        h, w = cv_img.shape[:2]

        detector.setInputSize((w, h))

        _, faces = detector.detect(cv_img)

        if faces is None:
            continue

        valid_faces = [
            f for f in faces if f[14] >= 0.7
        ]

        # ==================================
        # MULTIPLE FACE → SKIP
        # ==================================
        if len(valid_faces) > 1:

            return rotated, "SKIP", angle

        # ==================================
        # SINGLE FACE
        # ==================================
        if len(valid_faces) == 1:

            f = valid_faces[0]

            face_box = (
                int(f[0]),
                int(f[1]),
                int(f[2]),
                int(f[3])
            )

            # Landmark validation
            left_eye  = np.array([f[4],  f[5]])
            right_eye = np.array([f[6],  f[7]])
            nose      = np.array([f[8],  f[9]])
            mouth_l   = np.array([f[10], f[11]])
            mouth_r   = np.array([f[12], f[13]])

            eye_mid   = (left_eye + right_eye) / 2
            mouth_mid = (mouth_l + mouth_r) / 2

            if (
                nose[1] > eye_mid[1]
                and mouth_mid[1] > eye_mid[1]
                and right_eye[0] > left_eye[0]
            ):

                if angle != 0:
                    print(f"   🔄 Auto Rotated {angle}°")

                return rotated, face_box, angle

    return pil_img, None, 0

# ==========================================
# CROP
# ==========================================
def square_crop_no_stretch(pil_img, face_box):

    w, h = pil_img.size

    # ==================================
    # FACE CROP
    # ==================================
    if face_box:

        fx, fy, fw, fh = face_box

        cx = fx + fw // 2
        cy = fy + fh // 2

        side = int(fh * 3.0)

    # ==================================
    # CENTER CROP
    # ==================================
    else:

        cx = w // 2
        cy = h // 2

        side = min(w, h)

    left   = cx - side // 2
    top    = cy - int(side * 0.40)

    right  = left + side
    bottom = top + side

    # Boundary fixes
    if left < 0:

        left = 0
        right = side

    if right > w:

        right = w
        left = max(0, w - side)

    if top < 0:

        top = 0
        bottom = side

    if bottom > h:

        bottom = h
        top = max(0, h - side)

    return pil_img.crop(
        (left, top, right, bottom)
    )

# ==========================================
# KEEP ORIGINAL QUALITY
# ==========================================
def keep_original_quality(pil_img):

    return pil_img.resize(
        FINAL_SIZE,
        Image.Resampling.LANCZOS
    )

# ==========================================
# MAIN PROCESS
# ==========================================
print("\n🚀 Starting Processing...\n")

for filename in sorted(os.listdir(INPUT_FOLDER)):

    if not filename.lower().endswith(
        ('.jpg', '.jpeg', '.png')
    ):
        continue

    print(f"\n📸 Processing: {filename}")

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    img = load_and_fix_rotation(image_path)

    if img is None:
        continue

    # ==================================
    # BLUR CHECK
    # ==================================
    if is_blurry(img):

        print("⚠️ Blur image detected")

    # ==================================
    # FACE DETECTION
    # ==================================
    img, face_box, angle = detect_face_with_rotation(img)

    # ==================================
    # MULTIPLE FACE SKIP
    # ==================================
    if face_box == "SKIP":

        print("⏭️ Skipped: Multiple Faces")

        continue

    # ==================================
    # NO FACE
    # ==================================
    if face_box is None:

        print("➡️ No valid face found")
        print("➡️ Using center crop")

    else:

        print("✅ Face detected")

    # ==================================
    # CROP
    # ==================================
    cropped = square_crop_no_stretch(
        img,
        face_box
    )

    # ==================================
    # FINAL RESIZE
    # ==================================
    final = keep_original_quality(cropped)

    # ==================================
    # SAVE
    # ==================================
    out_name = (
        f"{os.path.splitext(filename)[0]}_400x400.jpg"
    )

    out_path = os.path.join(
        OUTPUT_FOLDER,
        out_name
    )

    final.save(
        out_path,
        "JPEG",
        quality=100,
        subsampling=0
    )

    print(f"✅ Saved: {out_name}")

print("\n✨ Processing Complete")
print(f"📁 Output Folder: {OUTPUT_FOLDER}")
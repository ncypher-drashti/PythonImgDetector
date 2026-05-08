import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# ==========================================
# INSTALL REQUIRED LIBRARIES
# ==========================================
# pip install pillow pillow-heif opencv-python numpy

# ==========================================
# FOLDERS
# ==========================================
INPUT_FOLDER  = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput_cropped"

FINAL_SIZE = (400, 400)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# CONVERT HEIC TO JPG
# ==========================================
def convert_heic_to_jpg(folder):

    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()

    except:
        print("❌ Install pillow-heif")
        print("pip install pillow-heif")
        return

    for filename in sorted(os.listdir(folder)):

        if not filename.lower().endswith(".heic"):
            continue

        heic_path = os.path.join(folder, filename)

        jpg_name = os.path.splitext(filename)[0] + "_converted.jpg"
        jpg_path = os.path.join(folder, jpg_name)

        if os.path.exists(jpg_path):
            continue

        try:
            img = Image.open(heic_path).convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)

            print(f"🔄 Converted: {filename}")

        except Exception as e:
            print(f"❌ Failed: {filename} -> {e}")

# ==========================================
# DOWNLOAD YUNET MODEL
# ==========================================
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx"
)

if not os.path.exists(MODEL_PATH):

    print("⬇️ Downloading YuNet model...")

    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    urllib.request.urlretrieve(url, MODEL_PATH)

    print("✅ Model downloaded")

# ==========================================
# LOAD FACE DETECTOR
# ==========================================
detector = cv2.FaceDetectorYN_create(
    MODEL_PATH,
    "",
    (320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=10
)

# ==========================================
# LOAD IMAGE WITH CORRECT ORIENTATION
# ==========================================
def load_image_correct_orientation(path):

    try:

        pil = Image.open(path)

        exif = pil.getexif()

        orientation = exif.get(274)

        if orientation == 3:
            pil = pil.rotate(180, expand=True)

        elif orientation == 6:
            pil = pil.rotate(270, expand=True)

        elif orientation == 8:
            pil = pil.rotate(90, expand=True)

        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        return img

    except:

        return cv2.imread(path)

# ==========================================
# FACE DETECTION + AUTO ROTATE
# ==========================================
def detect_face_yunet(img):

    rotations = [
        ("0", None),
        ("90", cv2.ROTATE_90_CLOCKWISE),
        ("180", cv2.ROTATE_180),
        ("270", cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    for angle_name, rotation in rotations:

        if rotation is not None:
            test_img = cv2.rotate(img, rotation)
        else:
            test_img = img.copy()

        H, W = test_img.shape[:2]

        detector.setInputSize((W, H))

        _, faces = detector.detect(test_img)

        if faces is None or len(faces) == 0:
            continue

        valid_faces = []

        for face in faces:

            score = face[14]

            if score < 0.7:
                continue

            valid_faces.append(face)

        # NO FACE
        if len(valid_faces) == 0:
            continue

        # MULTIPLE FACE -> SKIP
        if len(valid_faces) > 1:
            return test_img, "MULTIPLE"

        # SINGLE FACE
        face = valid_faces[0]

        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])

        return test_img, (x, y, w, h)

    return img, None

# ==========================================
# LITTLE ZOOM OUT CROP
# ==========================================
def crop_portrait(img, x, y, w, h):

    H, W = img.shape[:2]

    cx = x + w // 2
    cy = y + h // 2

    # LITTLE MORE ZOOM OUT
    pad_x   = int(w * 1.9)
    pad_top = int(h * 2.1)
    pad_bot = int(h * 3.2)

    x1 = max(cx - pad_x, 0)
    y1 = max(cy - pad_top, 0)
    x2 = min(cx + pad_x, W)
    y2 = min(cy + pad_bot, H)

    crop = img[y1:y2, x1:x2]

    return crop

# ==========================================
# NO UPSCALE
# ==========================================
def enhance_image(img):

    return img

# ==========================================
# CONVERT HEIC
# ==========================================
print("\n========== CONVERTING HEIC ==========")

convert_heic_to_jpg(INPUT_FOLDER)

# ==========================================
# PROCESS IMAGES
# ==========================================
print("\n========== PROCESSING IMAGES ==========")

results = []

SUPPORTED = ('.jpg', '.jpeg', '.png')

for filename in sorted(os.listdir(INPUT_FOLDER)):

    if not filename.lower().endswith(SUPPORTED):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)

    name, _ = os.path.splitext(filename)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{name}_400x400.jpg"
    )

    # ======================================
    # LOAD IMAGE
    # ======================================
    img = load_image_correct_orientation(input_path)

    if img is None:

        print(f"❌ Cannot read: {filename}")

        results.append((filename, "FAILED"))

        continue

    # ======================================
    # FACE DETECT
    # ======================================
    img, face = detect_face_yunet(img)

    # MULTIPLE FACE
    if face == "MULTIPLE":

        print(f"⚠️ SKIPPED {filename} -> Multiple faces")

        results.append((filename, "MULTIPLE FACE"))

        continue

    # SINGLE FACE
    if face:

        x, y, w, h = face

        cropped = crop_portrait(img, x, y, w, h)

        status = "OK - FACE"

    else:

        print(f"⚠️ No face: {filename}")

        H, W = img.shape[:2]

        size = min(W, H)

        x1 = max(W // 2 - size // 2, 0)

        cropped = img[0:size, x1:x1 + size]

        status = "FALLBACK"

    # ======================================
    # NO UPSCALE
    # ======================================
    cropped = enhance_image(cropped)

    # ======================================
    # FINAL 400x400
    # ======================================
    pil = Image.fromarray(
        cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    )

    pil = pil.resize(FINAL_SIZE, Image.LANCZOS)

    # ======================================
    # SAVE
    # ======================================
    pil.save(
        output_path,
        "JPEG",
        quality=95,
        subsampling=0
    )

    print(f"✅ {filename} -> {name}_400x400.jpg [{status}]")

    results.append((filename, status))

# ==========================================
# SUMMARY
# ==========================================
print("\n========== SUMMARY ==========")

for fname, status in results:

    print(f"{fname:30s} -> {status}")

ok = sum(1 for _, s in results if "FACE" in s)
fallback = sum(1 for _, s in results if "FALLBACK" in s)
multiple = sum(1 for _, s in results if "MULTIPLE" in s)
failed = sum(1 for _, s in results if "FAILED" in s)

print(f"\nTotal      : {len(results)}")
print(f"Face OK    : {ok}")
print(f"Fallback   : {fallback}")
print(f"Multiple   : {multiple}")
print(f"Failed     : {failed}")

print(f"\n✅ Output Folder:")
print(OUTPUT_FOLDER)
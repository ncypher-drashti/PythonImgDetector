import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
import urllib.request

# ==========================================
# INSTALL REQUIRED LIBRARIES
# ==========================================
# pip install pillow pillow-heif opencv-python numpy opencv-contrib-python

# ==========================================
# FOLDERS
# ==========================================
INPUT_FOLDER  = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput_cropped"

FINAL_SIZE = (400, 400)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# UPSCALE SETTINGS
# ==========================================
# Set to True to enable AI-based upscaling (requires opencv-contrib-python)
# Set to False to use high-quality Lanczos + sharpening fallback (always works)
USE_AI_UPSCALE = False   # Change to True if you have opencv-contrib-python installed

UPSCALE_FACTOR = 2       # How much to upscale before resizing to 400x400
                         # 2 = 2x upscale, 4 = 4x upscale

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
# LOAD AI UPSCALER (optional)
# ==========================================
sr_model = None

if USE_AI_UPSCALE:
    try:
        from cv2 import dnn_superres
        sr_model = dnn_superres.DnnSuperResImpl_create()

        # Download EDSR model if not present
        # Models: EDSR_x2, EDSR_x3, EDSR_x4, ESPCN_x2, FSRCNN_x2, LapSRN_x2
        sr_model_name = f"EDSR_x{UPSCALE_FACTOR}"
        sr_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{sr_model_name}.pb"
        )

        if not os.path.exists(sr_model_path):
            print(f"⬇️ Downloading SR model ({sr_model_name})...")
            sr_url = f"https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/{sr_model_name}.pb"
            urllib.request.urlretrieve(sr_url, sr_model_path)
            print("✅ SR model downloaded")

        sr_model.readModel(sr_model_path)
        sr_model.setModel("edsr", UPSCALE_FACTOR)
        print(f"✅ AI upscaler loaded: {sr_model_name}")

    except Exception as e:
        print(f"⚠️ AI upscaler not available: {e}")
        print("   Falling back to Lanczos + sharpening")
        sr_model = None

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
# UPSCALE IMAGE (NEW)
# ==========================================
def upscale_image(img_bgr):
    """
    Upscale a blurry/small image to improve quality before final resize.

    Two modes:
      1. AI upscaling (USE_AI_UPSCALE=True) — uses EDSR neural network
      2. Lanczos + sharpening (USE_AI_UPSCALE=False) — always available,
         no extra download needed, still much better than plain resize
    """

    H, W = img_bgr.shape[:2]

    # --- MODE 1: AI upscaling ---
    if USE_AI_UPSCALE and sr_model is not None:
        try:
            upscaled = sr_model.upsample(img_bgr)
            print(f"   🔬 AI upscale: {W}x{H} -> {upscaled.shape[1]}x{upscaled.shape[0]}")
            return upscaled
        except Exception as e:
            print(f"   ⚠️ AI upscale failed: {e}, using fallback")

    # --- MODE 2: Lanczos resize + Unsharp Mask sharpening ---
    # Step 1: Upscale with high-quality Lanczos interpolation
    new_W = W * UPSCALE_FACTOR
    new_H = H * UPSCALE_FACTOR

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_up = pil.resize((new_W, new_H), Image.LANCZOS)

    # Step 2: Unsharp Mask — brings back edge detail lost in blur
    # radius=2: how wide the sharpening halo is
    # percent=150: sharpening strength (100=subtle, 200=strong)
    # threshold=3: ignore tiny noise differences
    pil_sharp = pil_up.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Step 3: Slight contrast boost to make details pop
    enhancer = ImageEnhance.Contrast(pil_sharp)
    pil_final = enhancer.enhance(1.1)

    result = cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGB2BGR)
    print(f"   ✨ Upscaled: {W}x{H} -> {new_W}x{new_H} (Lanczos + Unsharp Mask)")

    return result

# ==========================================
# ENHANCE IMAGE  (kept + now calls upscaler)
# ==========================================
def enhance_image(img):
    """
    Previously a no-op. Now applies upscaling for blurry images.
    All original behavior preserved — upscaling happens before final resize.
    """
    img = upscale_image(img)
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
    # UPSCALE (was: no-op, now: active)
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
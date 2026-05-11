import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import urllib.request

# ==========================================
# SETTINGS
# ==========================================
INPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput_cropped"

FINAL_SIZE = (400, 400)

# Blur strictness
SHARP_VOTE_THRESHOLD = 3

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
# LOAD IMAGE
# ==========================================
def load_image(path):

    try:

        img = Image.open(path)

        # Fix mobile EXIF rotation
        img = ImageOps.exif_transpose(img)

        return img.convert("RGB")

    except Exception as e:

        print(f"❌ Error loading image: {e}")

        return None

# ==========================================
# DETECT FACES
# ==========================================
def detect_faces(pil_img):

    cv_img = cv2.cvtColor(
        np.array(pil_img),
        cv2.COLOR_RGB2BGR
    )

    h, w = cv_img.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(cv_img)

    if faces is None:
        return []

    return [f for f in faces if f[14] >= 0.7]

# ==========================================
# FIX UPSIDE DOWN ONLY
# ==========================================
def fix_upside_down(pil_img):

    # ---------- NORMAL ----------
    faces_0 = detect_faces(pil_img)

    valid_0 = []

    for f in faces_0:

        left_eye = np.array([f[4], f[5]])
        right_eye = np.array([f[6], f[7]])
        nose = np.array([f[8], f[9]])
        mouth_l = np.array([f[10], f[11]])
        mouth_r = np.array([f[12], f[13]])

        eye_mid = (left_eye + right_eye) / 2
        mouth_mid = (mouth_l + mouth_r) / 2

        if (
            nose[1] > eye_mid[1]
            and mouth_mid[1] > eye_mid[1]
            and right_eye[0] > left_eye[0]
        ):
            valid_0.append(f)

    # MULTIPLE FACES
    if len(valid_0) > 1:
        return pil_img, "SKIP"

    # SINGLE FACE
    if len(valid_0) == 1:

        f = valid_0[0]

        face_box = (
            int(f[0]),
            int(f[1]),
            int(f[2]),
            int(f[3])
        )

        return pil_img, face_box

    # ---------- UPSIDE DOWN ----------
    rotated = pil_img.rotate(
        -180,
        expand=True
    )

    faces_180 = detect_faces(rotated)

    valid_180 = []

    for f in faces_180:

        left_eye = np.array([f[4], f[5]])
        right_eye = np.array([f[6], f[7]])
        nose = np.array([f[8], f[9]])
        mouth_l = np.array([f[10], f[11]])
        mouth_r = np.array([f[12], f[13]])

        eye_mid = (left_eye + right_eye) / 2
        mouth_mid = (mouth_l + mouth_r) / 2

        if (
            nose[1] > eye_mid[1]
            and mouth_mid[1] > eye_mid[1]
            and right_eye[0] > left_eye[0]
        ):
            valid_180.append(f)

    # MULTIPLE FACES
    if len(valid_180) > 1:
        return rotated, "SKIP"

    # SINGLE FACE
    if len(valid_180) == 1:

        print("🔄 Upside down image fixed")

        f = valid_180[0]

        face_box = (
            int(f[0]),
            int(f[1]),
            int(f[2]),
            int(f[3])
        )

        return rotated, face_box

    return pil_img, None

# ==========================================
# FACE BLUR CHECK
# ONLY FACE IS CHECKED
# ==========================================
def is_face_blurry(pil_img, face_box):

    fx, fy, fw, fh = face_box

    pad = int(fh * 0.2)

    x1 = max(0, fx - pad)
    y1 = max(0, fy - pad)
    x2 = min(pil_img.width, fx + fw + pad)
    y2 = min(pil_img.height, fy + fh + pad)

    face = pil_img.crop((x1, y1, x2, y2))

    face = face.resize(
        (300, 300),
        Image.Resampling.LANCZOS
    )

    gray = cv2.cvtColor(
        np.array(face),
        cv2.COLOR_RGB2GRAY
    )

    # Method 1
    lap = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Method 2
    sobelx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobely = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel = np.sqrt(
        sobelx**2 + sobely**2
    ).mean()

    # Method 3
    tenengrad = (
        sobelx**2 + sobely**2
    ).mean()

    # Method 4
    brenner = np.sum(
        (
            gray[:-2, :].astype(float)
            - gray[2:, :].astype(float)
        ) ** 2
    ) / gray.size

    votes = sum([
        lap > 80,
        sobel > 8,
        tenengrad > 100,
        brenner > 50
    ])

    blurry = votes < SHARP_VOTE_THRESHOLD

    print(
        f"🔍 Face Quality → "
        f"Votes:{votes}/4 "
        f"{'⚠️ BLURRY' if blurry else '✅ SHARP'}"
    )

    return blurry

# ==========================================
# UPSCALE IMAGE
# ==========================================
def upscale_image(pil_img):

    img = cv2.cvtColor(
        np.array(pil_img),
        cv2.COLOR_RGB2BGR
    )

    upscaled = cv2.resize(
        img,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        upscaled,
        -1,
        kernel
    )

    return Image.fromarray(
        cv2.cvtColor(
            sharpened,
            cv2.COLOR_BGR2RGB
        )
    )

# ==========================================
# CROP FACE
# ==========================================
def crop_face(pil_img, face_box):

    w, h = pil_img.size

    fx, fy, fw, fh = face_box

    cx = fx + fw // 2
    cy = fy + fh // 2

    side = int(fh * 3.0)

    left = cx - side // 2
    top = cy - int(side * 0.40)

    right = left + side
    bottom = top + side

    # Boundary fix
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

    return pil_img.crop((left, top, right, bottom))

# ==========================================
# RESIZE
# ==========================================
def resize_final(img):

    return img.resize(
        FINAL_SIZE,
        Image.Resampling.LANCZOS
    )

# ==========================================
# CENTER CROP
# ==========================================
def center_crop(img):

    return ImageOps.fit(
        img,
        FINAL_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )

# ==========================================
# MAIN
# ==========================================
print("\n🚀 Starting Processing...\n")

for filename in sorted(os.listdir(INPUT_FOLDER)):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    print(f"\n📸 Processing: {filename}")

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    img = load_image(image_path)

    if img is None:
        continue

    # ======================================
    # FIX UPSIDE DOWN
    # ======================================
    img, face_box = fix_upside_down(img)

    # ======================================
    # MULTIPLE FACES
    # ======================================
    if face_box == "SKIP":

        print("⏭️ Multiple faces detected → skipped")

        continue

    # ======================================
    # NO FACE
    # CENTER CROP
    # ======================================
    if face_box is None:

        print("➡️ No face found → center crop")

        final = center_crop(img)

        out_name = (
            f"{os.path.splitext(filename)[0]}_center.jpg"
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

        print(f"✅ Saved center crop: {out_name}")

        continue

    print("✅ Face detected")

    # ======================================
    # FACE BLUR CHECK
    # ======================================
    blurry = is_face_blurry(
        img,
        face_box
    )

    # ======================================
    # BLURRY FACE
    # SAVE ORIGINAL ONLY
    # ======================================
    if blurry:

        print("⚠️ Blurry face → crop skipped")

        out_name = (
            f"{os.path.splitext(filename)[0]}_original.jpg"
        )

        out_path = os.path.join(
            OUTPUT_FOLDER,
            out_name
        )

        img.save(
            out_path,
            "JPEG",
            quality=100,
            subsampling=0
        )

        print(f"📋 Saved original image: {out_name}")

        continue

    # ======================================
    # SHARP FACE
    # UPSCALE + CROP
    # ======================================
    print("✨ Sharp face → upscale + crop")

    upscaled = upscale_image(img)

    fx, fy, fw, fh = face_box

    new_face_box = (
        fx * 2,
        fy * 2,
        fw * 2,
        fh * 2
    )

    cropped = crop_face(
        upscaled,
        new_face_box
    )

    final = resize_final(cropped)

    # ======================================
    # SAVE
    # ======================================
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
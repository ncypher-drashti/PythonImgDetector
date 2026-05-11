import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import urllib.request

# ==========================================
# SETTINGS
# ==========================================
INPUT_FOLDER = r"C:\Drashtiiii\python\images\imageinput"
OUTPUT_FOLDER = r"C:\Drashtiiii\python\images\image_blurry"
FINAL_SIZE = (400, 400)

# Lowered threshold to ensure even blurry images (like #12) get processed
SHARP_VOTE_THRESHOLD = 1 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# MODEL PATH & DOWNLOAD
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

download_file(
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    FACE_MODEL_PATH,
    "Face Detector"
)

detector = cv2.FaceDetectorYN_create(FACE_MODEL_PATH, "", (320, 320), score_threshold=0.6)

# ==========================================
# CORE FUNCTIONS
# ==========================================

def load_image(path):
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def detect_faces(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(cv_img)
    return [f for f in faces if f[14] >= 0.7] if faces is not None else []

def fix_upside_down(pil_img):
    """Detects face and checks orientation. Returns (image, face_box)."""
    for angle in [0, -180]:
        img_state = pil_img.rotate(angle, expand=True) if angle != 0 else pil_img
        faces = detect_faces(img_state)
        valid = []
        for f in faces:
            eye_mid_y = (f[5] + f[7]) / 2
            mouth_mid_y = (f[11] + f[13]) / 2
            if f[9] > eye_mid_y and mouth_mid_y > eye_mid_y: # Nose/Mouth below eyes
                valid.append(f)
        
        if len(valid) > 1: return img_state, "SKIP"
        if len(valid) == 1:
            if angle == -180: print("🔄 Upside down image fixed")
            return img_state, (int(valid[0][0]), int(valid[0][1]), int(valid[0][2]), int(valid[0][3]))
    
    return pil_img, None

def upscale_image_advanced(pil_img):
    """Advanced upscaling using Lanczos4 and Unsharp Masking for blurry inputs."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Step 1: High-quality Resize
    upscaled = cv2.resize(cv_img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

    # Step 2: Unsharp Mask (Better for reconstruction than a simple kernel)
    # This boosts edges without creating 'plastic' artifacts
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    sharpened = cv2.addWeighted(upscaled, 1.6, gaussian, -0.6, 0)

    # Step 3: Subtle Denoise to clean up artifacts from image #12
    final = cv2.fastNlMeansDenoisingColored(sharpened, None, 5, 5, 7, 21)

    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

def is_face_blurry(pil_img, face_box):
    fx, fy, fw, fh = face_box
    pad = int(fh * 0.2)
    face = pil_img.crop((max(0, fx-pad), max(0, fy-pad), min(pil_img.width, fx+fw+pad), min(pil_img.height, fy+fh+pad)))
    face = face.resize((300, 300), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    sobel = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1)**2).mean()
    
    votes = sum([lap > 70, sobel > 7]) # Relaxed thresholds
    blurry = votes < SHARP_VOTE_THRESHOLD
    print(f"🔍 Face Quality: {'⚠️ BLURRY' if blurry else '✅ SHARP'} (Votes: {votes})")
    return blurry

def crop_face(pil_img, face_box):
    w, h = pil_img.size
    fx, fy, fw, fh = face_box
    cx, cy = fx + fw // 2, fy + fh // 2
    side = int(fh * 3.2) # Slightly larger crop for better composition
    left, top = cx - side // 2, cy - int(side * 0.45)
    return pil_img.crop((max(0, left), max(0, top), min(w, left+side), min(h, top+side)))

# ==========================================
# MAIN EXECUTION
# ==========================================
print("\n🚀 Starting Heavy-Duty Processing...\n")

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")): continue
    
    print(f"📸 Processing: {filename}")
    img = load_image(os.path.join(INPUT_FOLDER, filename))
    if img is None: continue

    img, face_box = fix_upside_down(img)

    if face_box == "SKIP":
        print("⏭️ Multiple faces detected → skipped")
        continue

    if face_box is None:
        print("➡️ No face found → applying center crop")
        final = ImageOps.fit(img, FINAL_SIZE, centering=(0.5, 0.5))
        suffix = "_center"
    else:
        blurry = is_face_blurry(img, face_box)
        
        # Even if blurry, we process it now to help Image #12
        print("✨ Applying advanced upscaling + reconstruction...")
        upscaled = upscale_image_advanced(img)
        
        # Adjust face box for 2x upscale
        new_box = (face_box[0]*2, face_box[1]*2, face_box[2]*2, face_box[3]*2)
        cropped = crop_face(upscaled, new_box)
        final = cropped.resize(FINAL_SIZE, Image.Resampling.LANCZOS)
        suffix = "_reconstructed" if blurry else "_400x400"

    out_name = f"{os.path.splitext(filename)[0]}{suffix}.jpg"
    final.save(os.path.join(OUTPUT_FOLDER, out_name), "JPEG", quality=95, subsampling=0)
    print(f"✅ Saved: {out_name}")

print(f"\n✨ Complete! Check results in: {OUTPUT_FOLDER}")
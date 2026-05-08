import cv2
from PIL import Image
import numpy as np

def process_image(input_path, output_path):
    try:

        img = cv2.imread(input_path)

        if img is None:
            return "IMAGE_NOT_FOUND"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml'
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            cv2.imwrite(output_path, img)
            return "SUCCESS"

        x, y, w, h = faces[0]

        padding = int(w * 0.5)

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)

        x2 = min(x + w + padding, img.shape[1])
        y2 = min(y + h + padding, img.shape[0])

        cropped = img[y1:y2, x1:x2]

        final_img = cv2.resize(cropped, (400, 400))

        cv2.imwrite(output_path, final_img)

        return "SUCCESS"

    except Exception as e:
        return str(e)
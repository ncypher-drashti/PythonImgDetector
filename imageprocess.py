import cv2
import os
import time
from PIL import Image, ExifTags
import numpy as np

input_folder = "C:/Drashtiiii/CrickHunt_Web/src/Presentation/Nop.Web/wwwroot/images"
output_folder = "C:/Drashtiiii/python/output_images"

os.makedirs(output_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def fix_rotation(image):
    try:
        exif = image._getexif()
        if exif:
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_key = tag
                    break
            orientation = exif.get(orientation_key)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except:
        pass
    return image

def process_image(file_path):
    try:
        pil_image = Image.open(file_path)
        pil_image = fix_rotation(pil_image)
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return
        (x, y, w, h) = faces[0]
        center_x = x + w // 2
        center_y = y + h // 2
        crop_size = max(w, h) * 2
        start_x = max(center_x - crop_size // 2, 0)
        start_y = max(center_y - crop_size // 2, 0)
        end_x = min(start_x + crop_size, image.shape[1])
        end_y = min(start_y + crop_size, image.shape[0])
        cropped = image[start_y:end_y, start_x:end_x]
        final_image = cv2.resize(cropped, (500, 500))
        filename = f"{int(time.time() * 1000)}.jpg"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, final_image)
    except:
        pass

for file in os.listdir(input_folder):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(os.path.join(input_folder, file))
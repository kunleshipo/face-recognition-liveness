import cv2
import os
import sys
sys.path.append('/app')
from facetools import FaceDetection, IdentityVerification, LivenessDetection

SITE_ROOT = os.path.abspath(os.getcwd())
image_url = os.path.join(SITE_ROOT, "data", "images", "reynolds_003.png")

print(f"Reading image from {image_url}")

frame = cv2.imread(image_url)
if frame is None:
    raise ValueError("Failed to read image with OpenCV")

print("Image read successfully")

faceDetector = FaceDetection()
faces, boxes = faceDetector(frame)

if not len(faces):
    raise ValueError("No faces detected")
else:
    print(f"Detected {len(faces)} faces")


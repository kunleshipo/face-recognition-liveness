import os
from os import environ
from pathlib import Path
import sys
import subprocess

import cv2
import jsonpickle
import numpy as np
from dotenv import load_dotenv
from facetools import FaceDetection, IdentityVerification, LivenessDetection
from flask import Flask, Response, request
from PIL import Image
import logging
import io
from werkzeug.utils import secure_filename
from PyPDF2 import PdfFileReader
import fitz
import tempfile

import logging
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

root = Path(os.path.abspath(__file__)).parent.absolute()

load_dotenv((root / ".env").as_posix())  # take environment variables from .env.

data_folder = environ.get("DATA_FOLDER")
resnet_name = environ.get("RESNET")
deeppix_name = environ.get("DEEPPIX")
facebank_name = environ.get("FACEBANK")

data_folder = root.parent / data_folder

resNet_checkpoint_path = data_folder / "checkpoints" / resnet_name
facebank_path = data_folder / facebank_name

deepPix_checkpoint_path = data_folder / "checkpoints" / deeppix_name

faceDetector = FaceDetection()
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
)
livenessDetector = LivenessDetection(checkpoint_path=deepPix_checkpoint_path.as_posix())

app = Flask(__name__)

logger = logging.getLogger('werkzeug')
handler = logging.StreamHandler()
logger.addHandler(handler)

app.logger.setLevel(logging.DEBUG)

@app.route("/", methods=["GET"])
def index():
    response = {
        "message": "Liveness service is up and running"
    }

    logging.debug(f"Data folder: {data_folder}")
    logging.debug(f"ResNet: {resnet_name}")
    logging.debug(f"DeepPix: {deeppix_name}")
    logging.debug(f"Facebank: {facebank_name}")

    #SITE_ROOT = os.path.realpath(os.path.dirname(__file__))

    SITE_ROOT = os.path.abspath(os.getcwd()) #os.path.realpath(os.path.join(os.getcwd(), os.path.pardir))

    image_url = os.path.join(SITE_ROOT, "data","images", "reynolds_003.png")

    image = Image.open(image_url)

    logger.debug("Image Loaded")

    image_data = np.asarray(image)
    # convert string of image data to uint8
    nparr = np.frombuffer(image.tobytes(), np.uint8)

    logger.debug("Image Decoded")

    # decode image
    frame = cv2.imread(image_url)

    logger.debug("Image Frame Read")
    
    faces, boxes = faceDetector(frame)

    logger.debug("Checked Faces")

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "liveness_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        # add dummy file_name to the face_arr
        #face_arr = ['GCDE'] + face_arr
        min_sim_score, mean_sim_score, sim_index = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        response = {
            "message": "Everything is OK.",
            "liveness_score": liveness_score.item(),
        }
        status_code = 200
    
    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )

@app.route("/main", methods=["POST"])
def main():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "min_sim_score": None,
            "mean_sim_score": None,
            "liveness_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        response = {
            "message": "Everything is OK.",
            "min_sim_score": min_sim_score.item(),
            "mean_sim_score": mean_sim_score.item(),
            "liveness_score": liveness_score.item(),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )


@app.route("/identity", methods=["POST"])
def identity():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "min_sim_score": None,
            "mean_sim_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)

        response = {
            "message": "Everything is OK.",
            "min_sim_score": min_sim_score.item(),
            "mean_sim_score": mean_sim_score.item(),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )


@app.route("/liveness", methods=["POST"])
def liveness():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "liveness_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        response = {
            "message": "Everything is OK.",
            "liveness_score": liveness_score.item(),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )

@app.route("/liveness_mod", methods=["POST"])
def liveness_mod():
    r = request

    file = request.files['file']

    # convert string of image data to uint8
    nparr = np.frombuffer(file.read(), np.uint8) #np.frombuffer(r.data, np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "There is not any faces in the image.",
            "liveness_score": None,
        }
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)
        
        cut_off_score = 0.6
        if liveness_score >= cut_off_score:
            response = {
            "message": "Everything is OK.",
            "liveness_score": 0.95,
            }
            status_code = 200
        
        else:
            response = {
            "message": "Everything is OK.",
            "liveness_score": liveness_score.item(),
            }
            status_code = 200

        

    response_pickled = jsonpickle.encode(response)
    print(f'Liveness Score: {liveness_score}')
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )

@app.route("/verify_existing_graduated", methods=["POST"])
def verify_existing_graduated():
    r = request
    
    file = request.files['file']

    filename = secure_filename(file.filename)

    acceptable_file_types = ['pdf', 'jpg', 'jpeg']

    file_extension = filename.split('.')[-1]

    if file_extension not in acceptable_file_types:
        response = {
            "message": "File type is not acceptable."
        }
        status_code = 500
        response_pickled = jsonpickle.encode(response)
        return Response(
            response=response_pickled, status=status_code, mimetype="application/json"
        )
    
    image_files = []

    # if file is pdf, extract face images from the pdf
    if file_extension == 'pdf':
        with tempfile.TemporaryDirectory() as path:
            pdf_path = os.path.join(path, filename)
            file.save(pdf_path)

            pdf = PdfFileReader(pdf_path)
            pdf_images = []

            for page_num in range(pdf.getNumPages()):
                page = pdf.getPage(page_num)
                page_images = page.extract_images()
                for img in page_images:
                    pdf_images.append(img)

            for img in pdf_images:
                img_data = img['image']
                img_ext = img_data.info.get('Filter')
                img_ext = img_ext.split('/')[1]
                img_data = img_data.get_data()
                img_data = Image.open(io.BytesIO(img_data))
                img_data.save(os.path.join(path, f'{page_num}_{img_ext}'))

            image_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('jpg') or file.endswith('jpeg')]

    else:
        image_files.append(file)

    matched_faces = []

    for image_file in image_files:
        image = Image.open(image_file)
        image_data = np.asarray(image)
        # convert string of image data to uint8
        nparr = np.frombuffer(image.tobytes(), np.uint8)
        # decode image
        frame = cv2.imread(image_file)
        faces, boxes = faceDetector(frame)

        if len(faces):
            face_arr = faces[0] # Might have to append dummy file_name to the face_arr
            face_arr = face_arr.prepend('GCDE')
            min_sim_score, mean_sim_score, matched_filename = identityChecker(face_arr)
            #image_files_uint8.append(image_file)
            if min_sim_score < 1.0:  # Set a threshold for matching
                matched_faces.append({
                    "matched_filename": filename,
                    "similarity_score": min_sim_score
                })

    response = {
        "matched_faces": matched_faces
    }
    status_code = 200
    response_pickled = jsonpickle.encode(response)
    return Response(
        response=response_pickled, status=status_code, mimetype="application/json"
    )


if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0")

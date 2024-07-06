from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 1: Initialize ObjectDetector with appropriate options.
base_options = python.BaseOptions(model_asset_path='models/det/efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    content = await file.read()

    # STEP 2: Load the input image.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 3: Detect objects in the input image.
    detection_result = detector.detect(image)

    # STEP 4: Process the detection result.
    counts = len(detection_result.detections)
    person_count = 0
    object_list = []

    for detection in detection_result.detections:
        object_category = detection.categories[0].category_name
        object_list.append(object_category)
        if object_category.lower() == "person":
            person_count += 1

    return {"person_count": person_count,
            "object_list": object_list}

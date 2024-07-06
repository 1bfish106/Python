from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. 패키지를 가져옴
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 베이스 옵션은 모든곳에서 사용
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

from PIL import Image
import numpy as np
import io
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()

    # file.wirte(content, "./data/text.jpg") 는 쓰지마라

    #content -> jpg 파일인데 http통신에서는 파일이 character type으로 왔다갔다함
    #1 text -> binary  :io.BytesIO(text)
    #2 binary -> PIL image

    # STEP 3: Load the input image. 추론함
    # image = mp.Image.create_from_file(IMAGE_FILENAMES[3])
    binary = io.BytesIO(content)
    pil_img = Image.open(binary)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image. 추론결과를 받음 => 추론을 했기때문에 여기서 끝남
    # 사진은 비정형 데이터
    classification_result = classifier.classify(image)
    # print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    return {"result": result}
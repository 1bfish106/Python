# STEP 01 모델 가져옴
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


assert insightface.__version__>='0.3'

# STEP 02 추론기
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 03 
# 샘플 데이터 가져옴
from insightface.data import get_image as ins_get_image
img1 = cv2.imread("nam.jpg")
img2 = cv2.imread("na.jpg")


# STEP 04 추론하면 됨
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1
assert len(faces2)==1

# print(faces1[0])

# STEP 05 추론 완료~
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./na_output.jpg", rimg)

# then print all-to-all face similarity
# feats = []
# for face in faces1:
#     feats.append(face.normed_embedding)


feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sim = np.dot(feat1, feat2.T)
print(sim)


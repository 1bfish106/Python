from fastapi import FastAPI, Form

#step 1
from transformers import pipeline

#step 2
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")


app = FastAPI()


@app.post("/textClassification/")
async def login(text: str = Form()):

    #step 3
    # text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    # text = "똑같은 '직장인'인데, 우리는 왜 국민연금 보험료를 두배 내나요?"

    #step 4
    result = classifier(text)

    #step 5
    print(result)

    return {"result": text}
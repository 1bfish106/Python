# STEP 1
from transformers import pipeline

# STEP 2
classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")

# STEP 3
text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

# STEP 4
result = classifier(text)

print(result)
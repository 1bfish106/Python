from transformers import pipeline

text = "translate English to Korean: this is banana"

translator = pipeline("translation_xx_to_yy", model="facebook/my_awesome_opus_books_model")
translator(text)
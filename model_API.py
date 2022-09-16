from fastapi import FastAPI
from joblib import load
import regex as re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#from spamClassify import my_classifier



app = FastAPI()

filename = 'spam_model2'
model = load(filename)



def cleanText(text):
    text = re.sub(r"[\W_]",' ',text.lower()).replace('-','')
    words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    new_text = " ".join(words)
    return new_text

def classify(model,text):

    text = cleanText(text)
    prediction = model.predict([text])[0]
    res = 'Ham' if prediction == 0 else 'spam'
    spam_prob = model.predict_proba([text])[0][1]

    return {'label': res, 'spam_probability': float(spam_prob)*100}

@app.get('/')
def get_root():

	return {'message': 'Welcome to the SMS spam detection API'}


@app.get('/spam_detection_path/{message}')
async def detect_spam_path(message: str):
	return classify(model, message)

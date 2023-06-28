from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re, string
import nltk
from nltk.corpus import stopwords

model = pickle.load(open('music_genre_model-2.pkl', 'rb'))
tfidf = pickle.load(open('music_genre_vectorizer.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    # return "Hello world"
    return render_template('index.html')


def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'https?://\s+[www\.\st]', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = rem_stw(text)
    return text


def rem_stw(text):
    n_text = []
    for word in text.split():
        if word not in stopwords.words('english'):
            n_text.append(word)
    x = n_text[:]
    n_text.clear()
    return " ".join(x)


@app.route('/predict', methods=['POST'])
def prediction():
    lyrics = request.form.get('lyrics')

    # preprocess
    plyrics = preprocess(lyrics)
    vectorized = tfidf.transform([plyrics]).toarray()[0]

    # # prediction
    p = model.predict(np.expand_dims(vectorized, axis=0))
    result = 0
    if p[0][0] > 0.5:
        return lyrics+":   pop"
    return lyrics+":   rap"

    # return str(vectorized)


if __name__ == '__main__':
    app.run(debug=True)

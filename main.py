import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re

lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words=None)

folder_path = 'Data'
intents_json_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.json'):
            intents_json_files.append(os.path.join(root, file))

intents_json = []
for file in intents_json_files:
    with open(file, 'r') as f:
        intents_json.append(json.load(f))

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

def clean_up_sentence(sentence):
    if not isinstance(sentence, str):
        sentence = str(sentence)
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\\[a-zA-Z]+', '', sentence)
    return sentence

def predict_class(sentence, model, words, classes):
    sentence = clean_up_sentence(sentence)
    tokens = ' '.join(word_tokenize(sentence))
    vector = vectorizer.fit_transform([tokens])
    predictions = model.predict(vector)
    max_probability = np.max(predictions)
    max_probability_index = np.argmax(predictions)
    return classes[max_probability_index]

def get_response(ints, intents_json):
    intents_list = predict_class(ints, model, words, classes)
    if 'solution' in intents_list[0]:
        return intents_list[0]['solution']
    else:
        return 'I\'m sorry, I didn\'t understand that.'

print("Chatbot is up!")

while True:
    message = input("You :")
    ints = predict_class(message, model, words, classes)
    res = get_response(ints, intents_json)
    print(f"Bot : {res}")
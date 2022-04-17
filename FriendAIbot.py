import pickle
import nltk
from nltk import WordNetLemmatizer
import numpy as np
import random
import json
from tensorflow.keras.models import load_model
wnl = WordNetLemmatizer()
ignore = ['.', '!', '?', ',',"'"]
model = load_model('friend_model')
words = pickle.load(open('words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
with open('intents.json') as file:
    dicti = json.load(file)


def prepare_sentence(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = [wnl.lemmatize(word.lower()) for word in sentence if word not in ignore]
    return sentence


def create_bow(sentence):
    bow = []
    sentence = prepare_sentence(sentence)
    for word in words:
        bow.append(1) if word in sentence else bow.append(0)
    return np.expand_dims(np.array(bow),0)


def get_prediction(sentence):
    bow = create_bow(sentence)
    boc = model.predict(bow)
    tag = boc[0][0]
    for i in boc[0]:
        if i>tag:
            tag=i
    tagnum = list(boc[0]).index(tag)
    tag = tags[tagnum]
    responses = dicti['intents'][tagnum]['responses']
    return random.choice(responses)


while True:
    sentence = input("")
    print(get_prediction(sentence))
from nltk import WordNetLemmatizer
import json
import nltk
import random
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
wnl = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)
patterns = []
words = []
tags = []
docs = []
ignore = ['.', '!', '?', ',',"'"]
for dict in intents['intents']:
    for pattern in dict["patterns"]:
        patterns.append(pattern)
        docs.append([pattern,dict['tag']])
        wrds = nltk.word_tokenize(pattern)
        wrds2 = [word for word in wrds if word not in ignore]
        for word in wrds2:
            word = wnl.lemmatize(word)
        words.extend(wrds2)
    tags.append(dict['tag'])

random.shuffle(docs)
wordsfile = open('words.pkl','wb')
pickle.dump(words,wordsfile)
tagsfile = open('tags.pkl','wb')
pickle.dump(tags,tagsfile)
boc_empty = [0] * len(tags)
print(boc_empty)
training = []
for doc in docs:
    bow = []
    wrds = nltk.word_tokenize(doc[0])
    for word in words:
        bow.append(1) if word in wrds else bow.append(0)
    boc = list(boc_empty)
    boc[tags.index(doc[1])] = 1
    training.append([bow,boc])
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(input_shape=(len(words),),units = 32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=len(tags),activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
model.fit(x=np.array(train_x),y=np.array(train_y),epochs = 300,batch_size=1)
model.save('friend_model')


""

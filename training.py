import random
import json
import pickle
import numpy as np
import nltk
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model

lemmatizer = nltk.WordNetLemmatizer()

# Set the folder path and get all JSON files
folder_path = 'Data'
json_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(root, file))

# Initialize words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Read data from multiple JSON files
for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        if 'problem' not in data or 'type' not in data:
            print(f"Error: Missing 'problem' or 'type' key in file {file}")
            continue
        word_list = nltk.word_tokenize(data['problem'])
        if not word_list:
            print(f"Error: Empty word list in file {file}")
            continue
        documents.append((word_list, data['type']))
        if data['type'] not in classes:
            classes.append(data['type'])

print("Documents:", documents)

# Preprocess words and save to binary files
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Create and compile the model
inputs = Input(shape=(len(train_x[0]),))
fc1 = Dense(128, activation='relu')(inputs)
dropout1 = Dropout(0.5)(fc1)
fc2 = Dense(64, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(fc2)
outputs = Dense(len(train_y[0]), activation='softmax')(dropout2)

model = Model(inputs=inputs, outputs=outputs)

sgd = SGD(momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('chatbot.h5')
print("Model Saved and Trained")
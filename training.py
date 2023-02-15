import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# intents dosyasını yükle
with open(r'C:\Users\ÖMER KIYAK\Desktop\python\chatbot\intents.json', 'r') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Kelimeleri küçük harfe çevirin ve lemmatize edin
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

# Tekrarlanan kelimeleri çıkarın ve sıralayın
words = sorted(list(set(words)))

# Sınıf etiketlerini sıralayın
classes = sorted(list(set(classes)))

# Eğitim verileri için boş dizi
training_data = []
output_empty = [0] * len(classes)

# Training verilerini hazırla
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training_data.append([bag, output_row])

# Eğitim verilerini karıştırın ve Numpy dizisi olarak dönüştürün
random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)

# X ve y olarak ayrıştırın
train_x = np.array(list(training_data[:, 0]))
train_y = np.array(list(training_data[:, 1]))

# Yapay sinir ağı modeli oluşturma
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Modeli derleme
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Modeli eğit
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Eğitilen modeli kaydet
model.save('chatbot_model.h5', hist)

print("Model oluşturuldu")
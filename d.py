from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import gensim
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd

train = pd.read_csv('./twitter_training.csv', header=None)
train.columns = ['#', 'refers to', 'sentiment', 'text']
train = train[['text','sentiment']]
train["text"].isnull().sum()
train["text"].fillna("No content", inplace = True)

temp = []
data_to_list = train['text'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
list(temp[:5])
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
        

data_words = list(sent_to_words(temp))
def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)
data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
data = np.array(data)

labels = np.array(train['sentiment'])
y = []
for i in range(len(labels)):
    if labels[i] == 'Neutral':
        y.append(0)
    elif labels[i] == 'Negative':
        y.append(1)
    elif labels[i] == 'Positive':
        y.append(2)
    else:
        y.append(3)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 4, dtype="float32")
del y

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)

X_train, X_test, y_train, y_test = train_test_split(tweets,labels)
print (len(X_train),len(X_test),len(y_train),len(y_test))

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
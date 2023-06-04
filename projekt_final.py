from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import gensim
from sklearn.model_selection import train_test_split, KFold
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


def depure_data(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    data = re.sub('\S*@\S*\s?', '', data)

    data = re.sub('\s+', ' ', data)

    data = re.sub("\'", "", data)
        
    return data
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

X_fold, X_test, y_fold, y_test = train_test_split(tweets,labels, random_state=42)

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Define the number of folds and repetitions
num_folds = 5
num_repetitions = 2

# Define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store the evaluation results
all_scores = []

i = 1
# Perform cross-validation
for _ in range(num_repetitions):
    for train_index, test_index in kfold.split(X_fold):
        # Split the X_fold into training and test sets
        X_train, X_val = X_fold[train_index], X_fold[test_index]
        y_train, y_val = y_fold[train_index], y_fold[test_index]
        
        # Compile and train the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
        
        # Evaluate the model on the test set
        scores = model.evaluate(X_val, y_val, verbose=0)
        
        print(str(i)+". fold loss:",scores[0])
        print(str(i)+". fold accuracy:",scores[1])
        print(str(i)+". fold precision:",scores[2])
        print(str(i)+". fold recall:",scores[3])

        # Store the evaluation results
        all_scores.append(scores)
        i+=1

# Calculate the average scores across all repetitions and folds
all_scores = np.array(all_scores)
average_scores = np.mean(all_scores, axis=0)

# Print the average scores
print("Average loss:", average_scores[0])
print("Average accuracy:", average_scores[1])

final_score = model.evaluate(X_test, y_test, verbose=0)
print("Final loss:",final_score[0])
print("Final accuracy:",final_score[0])
print("Final precision:",final_score[0])
print("Final recall:",final_score[0])


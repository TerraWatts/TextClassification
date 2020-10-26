'''
Practice Exercise: Text classification with a neural network
Watts Dietrich
Oct 26 2020

The goal of this exercise is to use keras to build a neural network that will read movie reviews and estimate if
each review has a negative or positive sentiment.

I use the imdb dataset which is built in to the keras library. The dataset consists of 25,000 movie reviews that are
labeled as either positive (1) or negative (0). Each review has been encoded as a list of word indexes that take the
form of integers. The indices are ordered by word frequency, so the most frequently-encountered word is indexed as 1,
the second-most-frequently-encountered word is 2, etc.

A neural network is built and trained, then tested on validation data. An accuracy of roughly 87% is obtained.
In the accompanying TextClassification-SavingModels.py file, I save the trained model for later use and also
read in a raw .txt file of a movie review, process it, and use the saved model to predict its sentiment.
'''

import tensorflow as tf
from tensorflow import keras
import numpy as np

# load in the imdb movie dataset
data = keras.datasets.imdb

# create training and testing sets using the 10,000 most frequently-used words from the dataset
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# print the first entry in the dataset
# you will note that it displays as a list of integers
# this is a movie review where each word has been encoded as a different integer
print("The first review in the raw data form: ", train_data[0])

# get the integer-word mappings
word_index = data.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0  #padding will be added to make each review the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2  #this will be used in case of unknown words
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# make all reviews 250 words long. Longer reviews are cut down to size, shorter have the PAD tag added at the end
# this is important for setting the size of our neural network later
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding = "post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding = "post", maxlen=250)

# a function to convert the integer data to a human-readable review
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# decode the first review and display
print("The first review decoded: ", decode_review(test_data[0]))

# below, we build the model
# summary of how this works:
# the goal is to have a single output that shows whether the review is good or bad
# the embedding layer basically grants an ability to learn to recognize and group together words with similar meanings
#   it generates 10,000 16-dimensional word-vectors, one for each of the 10,000 words found in the dataset
#   as the model is trained, the angle between the vectors of similar words (e.g. "good" and "great") is reduced
# the GlobalAveragePooling1D layer reduces/scales-down the dimensionality of the 16D output of the Embedding layer,
#   decreases computational load, makes training easier
# the 16-node layer following the GAP1D layer will, hopefully, learn to classify various word patterns as good or bad
# finally, the output layer uses a sigmoid function to squish the output value to range from 0-1
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# split training data into training and validation sets for fitting model
# note there are 25000 total entries in the set, so here we will use 10000 for validation, 15000 for training
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# train the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# assess model accuracy
results = model.evaluate(test_data, test_labels)
print(results)

test_review = test_data[0]
predict = model.predict([test_review])
print("Sample Review: ")
print(decode_review(test_review))
print("Prediction: ", str(predict[0]))
print("Actual: ", str(test_labels[0]))

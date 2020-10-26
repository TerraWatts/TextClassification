Practice Exercise: Text classification with a neural network
Watts Dietrich
Oct 26 2020

The goal of this exercise is to use keras to build a neural network that will read movie reviews and estimate if
each review has a negative or positive sentiment.

I use the imdb dataset which is built in to the keras library. The dataset consists of 25,000 movie reviews that are
labeled as either positive (1) or negative (0). Each review has been encoded as a list of word indexes that take the
form of integers. The indices are ordered by word frequency, so the most frequently-encountered word is indexed as 1,
the second-most-frequently-encountered word is 2, etc.

In TextClassification.py, A neural network is built and trained, then tested on validation data. 
An accuracy of roughly 87% is obtained.

In the accompanying TextClassification-SavingModels.py file, I save the trained model for later use and also
read in a raw .txt file of a movie review, process it, and use the saved model to predict its sentiment.

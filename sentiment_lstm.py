# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D,Flatten
import keras.utils.np_utils as np_utils
from keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau

import random

from io import open
from nltk.probability import FreqDist
from matplotlib import pyplot as plt

# stop the training if the validation loss is increasing and the val top5 acc is no longer improving
earlyStoppingCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)
# user tensorboard for graphs
tensorBoardCallback = TensorBoard(log_dir="lstm_sentiment", histogram_freq=0, write_graph=True,
                                  write_images=False)
reduceOnPlateauCallback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0, mode='auto',
                                            epsilon=0.0001, cooldown=0, min_lr=0.00001)

callbacks = [earlyStoppingCallback, tensorBoardCallback, reduceOnPlateauCallback]

import tensorflow as tf


def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / ( precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)
# Embedding
#max_features = 20000

embedding_size = 300

# Convolution
filter_length = 3
nb_filter = 4
pool_length = 2


# LSTM

lstm_output_size = 4

# Training
batch_size = 100
nb_epoch = 20
test__train_split_ratio = 0.1# 0.1 means 10% test
nb_classes = 4
lr = 0.01

def corpus_to_indices(text):
    #words_map = build_words_map(text)

    #text_to_indices(text, words_map)
    global words_map
    # The vocabulary map

    words_map={}
    index=0


    # Initialize the output list
    text_indices = []
    maxlen = 0
    # Loop line by line
    for line in text:
        # Split into words
        line_words = line.split()

        if len(line_words) > maxlen:
            maxlen = len(line_words)
        # Initialize the line_indices
        line_indices = []
        # Loop word by word
        for word in line_words:
            # Store the word once in the wordMap
            if not word in words_map:
                words_map[word] = index
                # Increment the index for the next word
                index += 1

            # Add the index to the line_indices
            line_indices.append(words_map[word])

        # Add the line_indices to the output list
        text_indices.append(line_indices)


    return text_indices, len(words_map), maxlen

fd1=FreqDist()
def load_data(data_file_name, annotation_file_name):

    # Load text
    f_data = open(data_file_name, 'r', encoding='UTF8')

    text = []
    for line in f_data:
        text.append(line)
        words=line.split()
        for word in words:
          fd1[word] += 1


    text_indices, voc_size, maxlen = corpus_to_indices(text)

    # Load labels
    f_labels = open(annotation_file_name, 'r')

    labels = []
    for line in f_labels:
        labels.append(int(line) - 1)

    '''
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    labels = utils.to_categorical(encoded_Y)
    '''




    X_train, y_train, X_test, y_test = split_train_test(text_indices, labels)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, y_train, X_test, y_test, voc_size, maxlen

def split_train_test(corpus, labels):
    # Randomize the dataSet
    random_data = []
    random_labels = []

    # Sample indices from 0..len(corpus)
    size = len(corpus)
    rand_indices = random.sample(range(size), size)


    # Insert in the final dataset N=self.datasetSize random tweets from the rawData
    for index in rand_indices:
        random_data.append(corpus[index])
        random_labels.append(labels[index])

    # Calculate the test set size
    test_set_size = int(test__train_split_ratio * size)

    # The trainSet starts from the begining until before the end by the test set size
    train_set = random_data[0 : size - test_set_size]
    test_set  = random_data[len(train_set) : size]
    train_set_labels = random_labels[0 : size - test_set_size]
    test_set_labels  = random_labels[len(train_set) : size]
    return train_set, train_set_labels, test_set, test_set_labels


print('Loading data...')
data_file_name = "e:/sentiment_neural/tweets_data.txt"
lables_file_name = "e:/sentiment_neural/tweets_labels.txt"
X_train, y_train, X_test, y_test, max_features, maxlen = load_data(data_file_name, lables_file_name)


print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


import os
import codecs
import gensim
#aravec file
model = gensim.models.Word2Vec.load('e:/sata_neural/Twt-SG')
EMBEDDING_DIM=300

embedding_matrix = np.random.random((len(fd1.keys()) + 1, EMBEDDING_DIM))

#creating the embedding matrix
tweets_data=codecs.open(data_file_name,'r','utf-8')
tweets=tweets_data.readlines()
for tweet in tweets:
    words=tweet.split()
    count = 0.
    vec = np.zeros(300).reshape((1, 300))
    i=0
    for word in words:
        try:
            embedding_vector = model.wv[word]

            embedding_matrix[i] = embedding_vector
            i=i+1
        except KeyError:
            continue


model = Sequential()
#model.add(Embedding(max_features, embedding_size, input_length=maxlen))     #this line without using aravec
model.add(Embedding(max_features+1, embedding_size, input_length=maxlen,weights=[embedding_matrix]))    #this using aravec

model.add(Dropout(0.7))   #to avoid overfitting as the data is not that big as the data is not that big


# model.add(Convolution1D(nb_filter=nb_filter,
#                         filter_length=filter_length,
#                         border_mode='valid',
#                         activation='relu',
#                         subsample_length=1))
# model.add(MaxPooling1D(pool_length=pool_length))
#
#
# model.add(Convolution1D(nb_filter=nb_filter,
#                         filter_length=filter_length,
#                         border_mode='valid',
#                         activation='relu',
#                         subsample_length=1))
# model.add(MaxPooling1D(pool_length=pool_length))

#model.add(Flatten())

# model.add(Dropout(0.6)
# model.add(LSTM(4,activation='linear',return_sequences=True))    #if you wanna remove lstm layers,remove from the top i.e here,return_Sequence =true mean it is followed by another one*]
# model.add(Dropout(0.5))
# model.add(LSTM(4,activation='linear',return_sequences=True))
model.add(LSTM(5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

import keras.optimizers
opt = keras.optimizers.adagrad(lr)

model.compile(loss='categorical_crossentropy',
               optimizer=opt,
               metrics=['accuracy',f1_score])

print('Train...')
result=model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
         validation_data=(X_test, y_test),callbacks=callbacks)


model.save('atb_model', overwrite=True)
score, acc,f1_sc = model.evaluate(X_test, y_test, batch_size=batch_size)



print('Test loss:', score)
print('Test accuracy:', acc)
print('Test F1_score',f1_sc)
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Loss vs. Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('./graphs/loss_vs_epochs.png')

plt.plot(result.history['acc'])
plt.title('Accuracy vs. Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('./graphs/accuracy_vs_epochs.png')





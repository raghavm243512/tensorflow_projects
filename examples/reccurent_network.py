from keras.datasets import imdb # example dataset
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

# Each review is encoded by integers that represents how common a word is in the entire dataset. 
# For example, a word encoded by the integer 3 means that it is the 3rd most common word in the dataset.

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

# can not pass different length data into the network
train_data = sequence.pad_sequences(train_data, MAXLEN) # trim to 250
test_data = sequence.pad_sequences(test_data, MAXLEN) # or add 0 as padding for more values

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), # Embedding layer creates vector of dimension 32 from inputs. 
    tf.keras.layers.LSTM(32), # Each word has 32 dimensions associated with it. LSTM = Long-short term memory, keeps track of previous states
    tf.keras.layers.Dense(1, activation="sigmoid") # Classify 0-0.5 as bad, > 0.5 as good
])

model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc']) # Binary classification, optimizer doesn't matter much
history = model.fit(train_data, train_labels, epochs=3, validation_split=0.2) # use 20% of training data as validation
results = model.evaluate(test_data, test_labels)
print(results)
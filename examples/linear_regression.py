from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra library
import pandas as pd # data management
import matplotlib.pyplot as plt # plot data
from IPython.display import clear_output
from six.moves import urllib

from tensorflow import feature_column as fc # assigns input parameters

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived') # remove column 'survived' and store in y_train
y_eval = dfeval.pop('survived')
# dftrain.head() # view the first few entries in data
# dftrain.describe() # give data overview, count, mean, quartiles, std dev, etc.
# dftrain.age.hist(bins=20) # histogram of age, 20 bins
# dftrain.sex.value_counts().plot(kind='barh') # horizontal bar graph, categorical data
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive') # % surivors of each sex

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck','embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) 
  # adds feature column of categorical data with all possible feature values to tensorflow 

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32)) # adds numerical features

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32): # input data, output data, number of iterations, will data be rearranged, 
    #size of data portions fed
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # store a function for training
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False) # store a function for evaluation

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) # store a linear classifier, give it the features & potential values of features

linear_est.train(train_input_fn)  # train estimator, passing in a method which returns a tf.data.Dataset object
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

# clear_output()  # clears consoke output
print(result)
print(dfeval.loc[0]) # gets inputs of first entry in dataset
print(list(linear_est.predict(eval_input_fn))[0]['probabilities']) # call predict, give it a Dataset, getting probabilities of potential outputs from first item
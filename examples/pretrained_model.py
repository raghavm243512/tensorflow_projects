import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

keras = tf.keras

(raw_train, raw_validation, raw_test), metadata = tfds.load( # example image dataset, split into train, test, etc.
  'cats_vs_dogs',
  split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], # split all availible data into 3 sets
  with_info=True,
  as_supervised=True,
) # reference documentation to load example datasets
get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  # returns an image that is reshaped to IMG_SIZE
  image = tf.cast(image, tf.float32) # convert pixels to float.32 values
  image = (image/127.5) - 1 # makes pixel values range from -1 to 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example) # apply the function to every sample in the set
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3) # 160 x 160 x 3 imput base

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, # don't include the classifier from the premade network
                                               weights='imagenet') # get preloaded model values and weights from imagenet
base_model.summary()

# for image, _ in train_batches.take(1): 
#    pass
# feature_batch = base_model(image) # gives shape of last output layer
# print(feature_batch.shape)

base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() # converts a 5 x 5 x 1280 into a 1 x 1280 by averaging the values in each 5 x 5
prediction_layer = keras.layers.Dense(1) # single neuron

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # dual classification, loss is binary
              metrics=['accuracy'])

initial_epochs = 3
validation_steps=20

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
new_model = tf.keras.models.load_model('dogs_vs_cats.h5') # .h5 is a keras model file extension
# call .predict() on a model to run predictions
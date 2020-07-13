import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training
# labels are values from 0-9, each number representing a different clothing article
# images are 28x28 grayscales of various articles

# train_images.shape # lets us view amount of data & dimensions
# train_images[0,23,23] # access the 1st image, row & col @ index 23

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # different types of clothing in set

# display one of the training images
plt.figure() # make figure
plt.imshow(train_images[15]) # show image
plt.colorbar() # sidebar
plt.grid(False) # no grid
plt.show() # show created plot

# preprocess data. Adjusting values to be between 0-1 will work better on the default initialization of the network. 
# divide all values by 255.0 to create a decimal between 0 and 1 for grayscale pixels
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([ # keras has prebuilt network models. Sequential is the typical left to right processing network. Other types include convolutional, etc.
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1). flatten takes in a complex matrix shape and converts into a single column. In this case, could be
    # replaced with 784, but in other contexts, it's easier to map input shape to a flat column
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2) Dense means fully connected to previous layer. 128 neurons, activation is relu, but could be sigmoid
    # arctan, etc.
    # keras.layers.Dense(128, activation='relu'), # add extra layers like this
    keras.layers.Dense(10, activation='softmax') # output layer (3) 10 neurons, 10 outputs. Softmax activation specifially requires that all outputs add up to 1
    # Depending on context, this activation can be different like the previous one. 
]) # creating the model created the network structure itself

model.compile(optimizer='adam', # Vast majority of cases adam is used. It's the type of gradient descent/optimization taking place (batch, stochastic, etc)
              loss='sparse_categorical_crossentropy', # The loss function being optimized
              metrics=['accuracy']) # what we are using to evaluate the network. In classification it's accuracy, can vary

model.fit(train_images, train_labels, epochs=6)  # pass the data, labels and epochs, trains network. Too many epochs leads to overfitting

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) # evaluate returns loss and accuracy. verbose = 1 displays a progress bar
print('Test accuracy:', test_acc)

predictions = model.predict(test_images) # gives an array of predictions, with varying amounts assigned to 0-9
print(class_names[np.argmax(predictions[15])])
print(class_names[test_labels[15]])
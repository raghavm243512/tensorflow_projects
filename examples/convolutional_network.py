import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator # Data Augmentation. Messes with images to create more training data

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() # preloard data w/ RGB
train_images, test_images = train_images / 255.0, test_images / 255.0 # Assign color values to number between 0-1

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'] # different labels
IMG_INDEX = 7 # Image to view
plt.imshow(train_images[IMG_INDEX]) # display image
plt.xlabel(class_names[train_labels[IMG_INDEX][0]]) # X axis label
plt.show()

model = models.Sequential() # Left to right network
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # 32 filters, 3x3 size. Input shape of 32 x 32 x 3 for RGB and imazge size
model.add(layers.MaxPooling2D((2, 2))) # combines multiple inputs from previous layer into smaller dimensional data. For speed reasons
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # 64 filters, 3x3 size, input size is inferred
model.add(layers.MaxPooling2D((2, 2))) # due to pooling, the dimension of the data decreases, and thus we can add more depth (64 filters instead of 32)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) # convert multidimensional matrix from previous layer into a column
model.add(layers.Dense(64, activation='relu'))  # fully connected layer
model.add(layers.Dense(10)) # fully connected, 10 possible outputs, 10 layers
# convulutional layers extract features, fully connected layers interpret them
# model.summary() # displays traits of the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Loss function varies depending on task. from_logits = true means output layer isn't softmax
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images,  test_labels)
print(test_acc)

# datagen = ImageDataGenerator(
# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,
# fill_mode='nearest')

# # pick an image to transform
# test_img = train_images[20]
# img = image.img_to_array(test_img)  # convert image to numpy arry
# img = img.reshape((1,) + img.shape)  # reshape image

# i = 0

# for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
#     plt.figure(i)
#     plot = plt.imshow(image.img_to_array(batch[0]))
#     i += 1
#     if i > 4:  # show 4 images
#         break

# plt.show()
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color='blue'
    else:
        color='red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

im1 = Image.open("clothes_1.jpg")
im2 = Image.open("clothes_2.jpg")
im3 = Image.open("clothes_3.jpg")

im1_fix = im1.draft("L", (28,28))
im2_fix = im2.draft("L", (28,28))
im3_fix = im3.draft("L", (28,28))

im1_fix = im1.resize((28,28))
im2_fix = im2.resize((28,28))
im3_fix = im3.resize((28,28))

im1.show("im1 original")
im1_fix.show("im1 modified")
im2.show("im2 original")
im2_fix.show("im2 modified")
im3.show("im3 original")
im3_fix.show("im3 modified")

im1_fix.save("im1_fix.jpg")
im2_fix.save("im2_fix.jpg")
im3_fix.save("im3_fix.jpg")

np_im1 = numpy.array(im1_fix)
np_im2 = numpy.array(im2_fix)
np_im3 = numpy.array(im3_fix)

train_images =  train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

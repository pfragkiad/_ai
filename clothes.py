from __future__ import annotations #list[str] support

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

data = tf.keras.datasets.fashion_mnist.load_data()
#60k training, 10k testing
(train_images, train_labels), (test_images, test_labels) = data
print(train_images.shape) #(60000, 28, 28)
print(train_labels.shape) #(60000,)
print(test_images.shape) #(10000, 28, 28)
print(test_labels.shape) #(10000, )

#from typing import TypeAlias
#stringList : TypeAlias = list[str]

class_names : 'list[str]' = ["T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot"]

#images are stored as 28x28 numpy arrays, pixels from 0 to 255
#labels array of integers from 0 to 9

#https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download

#original dataset here:
#https://github.com/zalandoresearch/fashion-mnist

#print (tf.__version__)

print("Labels in range: ", np.min(train_labels), np.max(train_labels)) #0-9
print("Pixels in range: ", np.min(train_images), np.max(train_images)) #0-255

train_images_normal =  train_images/255.0
test_images_normal = test_images/255.0

plt.figure(figsize = (17,8))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images_normal[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

import os
file : str = 'model_01.dat'
model : tf.keras.Model 
if os.path.exists(file):
    model = tf.keras.models.load_model(file)
else:
    from keras import layers
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape= train_images.shape[1:])) #28, 28, = 784 output
    model.add(layers.Dense(128,activation = "relu"))
    model.add(layers.Dense(10, activation= 'softmax')) #if from_logits=true then softmax is not needed here
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])
    #feed
    model.fit(train_images_normal, train_labels, epochs = 10) #not a lot of epochs to avoid overfitting
    model.save(file)

model.summary()
test_loss, test_accuracy = model.evaluate(test_images_normal, test_labels, verbose = 2)
print(f'Test accuracy: {test_accuracy:.1%}')
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

# plt.figure(figsize = (17,8))
# for i in range(32):
#     plt.subplot(4,8,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images_normal[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

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

#see results (returns 10 numbers per test image!)
#np.ndarray
predictions = model.predict(test_images_normal)
print(type(predictions))

import heapq
#prediction confidences
predictionPercentages = predictions[0]
print(predictionPercentages)
largestValues = heapq.nlargest(2,predictionPercentages)
print(f'Real value: {class_names[test_labels[0]]}')
#print(sum(predictionPercentages))
for m in largestValues:
    i = np.where(predictionPercentages==m)[0][0]
    print(f'Predicted value: {class_names[i]} with probability {m:.01%}') 


predictedValue = class_names[np.argmax(predictionPercentages)]
print(f"Best prediction: {predictedValue}")

plt.figure()
for i in range(16):
    plt.subplot(4,8,2*i+1)
    plt.xticks([])
    plt.yticks([])
    bestPrediction : int = np.argmax(predictions[i])
    bestPredictionPercentage: float = np.max(predictions[i])
    color = 'blue' if bestPrediction == test_labels[i] else 'red'
    plt.imshow(test_images_normal[i], cmap=plt.cm.binary)
    plt.xlabel(f'{class_names[bestPrediction]} {bestPredictionPercentage:.1%} ({class_names[test_labels[i]]})')
    plt.subplot(4,8,2*i+2)
    #plt.xticks(range(10))
    plt.xticks(range(10),class_names,rotation=90)
    plt.yticks([])
    barPlot = plt.bar(range(10), predictions[i], color='grey')
    plt.ylim([0,1])
    barPlot[bestPrediction].set_color('red')
    barPlot[test_labels[i]].set_color('blue')

plt.show()

#insert a new axis with expand_dims
#add a single image to the batch
img = np.expand_dims(test_images[50],0) # (28,28) to (1, 28, 28)
#print(img.shape)
predictions_img = model.predict(img)
max_confidence = np.argmax(predictions_img)
print(f"Prediction: {class_names[max_confidence]}, Real: {class_names[test_labels[50]]}")

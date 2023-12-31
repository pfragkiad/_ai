import numpy as np

import keras.models
from keras.models import Sequential

import keras.layers as l
#from keras.layers import Dense

import keras.activations as activations
import keras.metrics as metrics
import keras.optimizers as optimizers
import keras.losses as losses

data = np.loadtxt("and_data.txt",dtype = np.int8, delimiter=' ')
samplesCount = len(data)

#count = data.shape[0]

#print(data.shape) #(4,3)
X = data[:,:-1]
y = data[:,-1].reshape((samplesCount,1))

#X = np.repeat(X,50,axis=0)
#y = np.repeat(y,50,axis=0)

featuresCount = X.shape[1]
samplesCount = X.shape[0]

import models
model : keras.models.Sequential  = models.getDetailedModel(featuresCount, 4)
# model.get_layer(name = 'First layer')
# model.get_layer(index=0)
# model.pop()

import tensorflow as tf

def custom_loss(y_true,y_pred):
    sq = tf.square(y_true-y_pred)
    return tf.reduce_mean(sq,axis=-1) #same as mse
#model.compile(optimizer='RMSprop',loss=custom_loss, metrics = ['accuracy'])
 
model.compile(optimizer='RMSprop',loss='binary_crossentropy', metrics = ['accuracy'])


# model.compile(
#     optimizer='RMSprop',
#     loss = losses.binary_crossentropy,
#     metrics = metrics.accuracy)

model.fit(X,y, epochs = 200) #@10 epoch the model is perfect
loss, acc = model.evaluate(X,y, verbose=0)

model.summary()
w = model.get_weights();
print(len(w))
print(w)
print(f'The accuracy is: {acc: .1%}')

# print(X)
# print(Y)
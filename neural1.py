import numpy as np

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
model  = models.getDetailedModel(featuresCount, 4)

model.compile(optimizer='RMSprop',loss='binary_crossentropy', metrics = ['accuracy'])
# model.compile(
#     optimizer='RMSprop',
#     loss = losses.binary_crossentropy,
#     metrics = metrics.accuracy)

model.summary()

model.fit(X,y, epochs = 200) #@10 epoch the model is perfect

loss, acc = model.evaluate(X,y, verbose=0)

w = model.get_weights();
print(len(w))
print(w)
print(f'The accuracy is: {acc: .1%}')

# print(X)
# print(Y)
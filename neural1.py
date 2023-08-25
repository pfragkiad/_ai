import numpy as np

from keras.models import Sequential
from keras.layers import Dense

import keras.layers.activation as activations
import keras.metrics as metrics
import keras.optimizers as optimizers
import keras.losses as losses

data = np.loadtxt("and_data.txt",dtype = np.int8, delimiter=' ')
count = len(data)

#count = data.shape[0]

#print(data.shape) #(4,3)
X = data[:,:-1]
x_columns = X.shape[1]
y = data[:,-1].reshape((count,1))


model = Sequential()

model.add(Dense(count , input_shape = (x_columns,))) #after the first layer you don't need to specify the size
#model.add(Dense(1,activation='sigmoid'))
model.add(Dense(1,activation=activations.sigmoid))
#model.compile(optimizer='RMSprop',loss='binary_crossentropy', metrics = 'accuracy')
model.compile(
    optimizer = optimizers.rmsprop,
    loss = losses.binary_crossentropy,
    metrics = metrics.binary_accuracy)
model.fit(X,y, epochs = 200)

loss, acc = model.evaluate(X,y, verbose=0)
print(f'The accuracy is: {acc: .1%}')

# print(X)
# print(Y)
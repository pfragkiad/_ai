
import keras.models as models
import keras.layers as layers

import keras.activations as activations

def getCompactModel(featuresPerSample : int,  nodesCount: int) -> models.Model :
    model = models.Sequential()
    model.add(layers.Dense(nodesCount , input_shape = (featuresPerSample,))) #after the first layer you don't need to specify the size
    #model.add(Dense(1,activation='sigmoid'))
    model.add(layers.Dense(1,activation=activations.sigmoid))
    return model

def getDetailedModel(featuresPerSample : int,  nodesCount: int) -> models.Model :
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape = (featuresPerSample,),))
    model.add(layers.Dense(nodesCount))
    model.add(layers.Dense(1))
    model.add(layers.Activation(activation=activations.sigmoid))

    return model
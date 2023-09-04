import tensorflow as tf

c = tf.add([1,3],[5,7])
print(c) #tf.Tensor([ 6 10], shape=(2,), dtype=int32)

import numpy as np

z1 = np.zeros((4,4))
z1 = np.expand_dims(z1,1)  #4,4 to 4,1,4
#z1 = np.expand_dims(z1,2)  #4,4 to 4,4,1
#z1 = np.expand_dims(z1) #error
#z1 = np.expand_dims(z1,3) #out of bounds
print(z1.shape)


#print(type(c))


print(tf.reduce_mean([1.0,2.0],axis=None))
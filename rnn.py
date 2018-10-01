import tensorflow as tf 

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import BatchNormalization
from keras.utils import np_utils

import time


np.random.seed(2) #makes sure random numbers are the same each runtime so results next time you run the code

cifar10 = tf.keras.datasets.cifar10.load_data() #the entire dataset

(X_train, y_train), (X_test, y_test) = cifar10  #training and test set, labels in y_
X_train = np.asarray(X_train, dtype=np.float32)/255.
y_train = np.asarray(y_train, dtype=np.int32)
X_test = np.asarray(X_test, dtype=np.float32)/255.
y_test = np.asarray(y_test, dtype=np.int32)

mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
X_train = (X_train - mean_pixel) / std_pixel
X_test = (X_test - mean_pixel) / std_pixel


n = X_train.shape[0] #50000
n_test =  X_test.shape[0]

names = ["airplane","automobile","bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_labels = len(names)

X_train = X_train.reshape((n,32,32*3))
X_test = X_test.reshape((n_test,32,32*3))
y_train =np_utils.to_categorical(y_train, num_labels)
y_test = np_utils.to_categorical(y_test, num_labels)

permutation = np.random.permutation(np.arange(n,dtype=np.int32))
X_train = X_train[permutation]
y_train = y_train[permutation]
valid_length = 1000 #keep it smaller than, what, 2k?
cursor = n-valid_length
X_valid = X_train[cursor:]
y_valid = y_train[cursor:]
X_train = X_train[:cursor]
y_train = y_train[:cursor]
batch = 128
epochs = 8+8
enable = True
hidden_nodes_lstm = 32*3**2
hidden_fc = 32*32*3

# create and fit the LSTM network
#Epochs = [2,3,4,6,7,8,9]
enabled = [0,1]
s = []
#for enable in enabled:
#    for epochs in Epochs:

model = Sequential()
model.add(BatchNormalization())
if enable:
    model.add(CuDNNLSTM(hidden_nodes_lstm, input_shape=X_train.shape[1:]))
else:
    model.add(LSTM(hidden_nodes_lstm, input_shape=X_train.shape[1:]))
model.add(Dropout(0.8))
model.add(Dense(hidden_fc,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_labels,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= 'adam',metrics=['accuracy'])
start = time.time()
model.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=1)#,validation_data=(X_valid, y_valid))
end = time.time()
scores = model.evaluate(X_test, y_test, verbose=0)
print scores
TIME = end - start
s.append(TIME)

print TIME, "sec, is the excecution time of %d epochs with cuda lstm enabled"%epochs
''' #takes the time measurement and saves it to file
with open('measurements.txt','a') as f:
    for i in range(len(s)):
        f.write('time %f epoch %d \n'%(s[i],Epochs[i%(len(Epochs))]))
'''
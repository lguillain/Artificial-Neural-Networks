import numpy as np
import time
import scipy.io

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD, Adam

import json

# you may experiment with different subsets, 
# but make sure in the submission 
# it is generated with the correct random seed for all exercises.
student1 = 285467
student2 = 233984
np.random.seed(hash(student1 + student2) % 2**32)
subset_of_classes = np.random.choice(range(10), 5, replace = False)

#--------------------

# convert RGB images x to grayscale using the formula for Y_linear in https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
def grayscale(x):
    x = x.astype('float32')/255
    x = np.piecewise(x, [x <= 0.04045, x > 0.04045], 
                        [lambda x: x/12.92, lambda x: ((x + .055)/1.055)**2.4])
    return .2126 * x[:,:,0,:] + .7152 * x[:,:,1,:]  + .07152 * x[:,:,2,:]

def downsample(x):
    return sum([x[i::2,j::2,:] for i in range(2) for j in range(2)])/4

def preprocess(data):
    gray = grayscale(data['X'])
    downsampled = downsample(gray)
    return (downsampled.reshape(16*16, gray.shape[2]).transpose(),
            data['y'].flatten() - 1)


data_train = scipy.io.loadmat('/housenumbers/train_32x32.mat')
data_test = scipy.io.loadmat('/housenumbers/test_32x32.mat')

x_train_all, y_train_all = preprocess(data_train)
x_test_all, y_test_all = preprocess(data_test)

def extract_classes(x, y, classes):
    indices = []
    labels = []
    count = 0
    for c in classes:
        tmp = np.where(y == c)[0]
        indices.extend(tmp)
        labels.extend(np.ones(len(tmp), dtype='uint8') * count)
        count += 1
    return x[indices], labels

x_train, y_train = extract_classes(x_train_all, y_train_all, subset_of_classes)
x_test, y_test = extract_classes(x_test_all, y_test_all, subset_of_classes)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#--------------------

class AppendedHistory:
    def __init__(self, varname=None, history_=None, fname=None):
        if history_:
            self.history = history_
        elif fname:
            with open(fname, 'r') as file:
                self.history = json.load(file)
        else:
            self.history = {
                'loss': [],
                'val_loss': [],
                'acc': [],
                'val_acc': [],
                'var': []
            }
        self.varname = varname
        
    def append_hist(self, var, history):
        """Append fitting history for a new variable."""
        for key in self.history.keys():
            if key is not 'var':
                self.history[key].append(history.history[key])
            else:
                self.history['var'].append(var)
        
    def add_hist(self, var, history):
        """Add further fitting history to a certain variable."""
        ind = self.history['var'].index(var)
        for key in self.history.keys():
            if key is not 'var':
                self.history[key][ind] = np.concatenate((self.history[key][ind], history.history[key])).tolist()
        
    def save(self, fname):
        with open(fname, 'w') as file:
            file.write(json.dumps(self.history))

#--------------------

for i in range(3, 11):
    apphist_layers = AppendedHistory(varname='weights')
    for j in range(10):
        one_hl = Sequential()
        one_hl.add(Dense(2**i, input_dim=x_train.shape[1], activation='relu'))
        one_hl.add(Dense(5, input_dim=x_train.shape[1], activation='softmax'))
        np.save('/output/weights_{}-{}'.format(2**i, j), one_hl.get_weights())
        one_hl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        hist_one_hl_layers = one_hl.fit(x_train, y_train, epochs=300, batch_size=128, \
        									  verbose=0, validation_data=(x_test, y_test))
        apphist_layers.append_hist(j, hist_one_hl_layers)
    apphist_layers.save('/output/apphist_layers_{}.txt'.format(2**i))


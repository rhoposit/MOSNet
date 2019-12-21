import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm

class CNN(object):
    
    def __init__(self, dims):
        print('CNN init')
        self.dims = dims
        
    def build(self):
        _input = keras.Input(shape=(None, self.dims))
        
        re_input = layers.Reshape((-1, self.dims, 1), input_shape=(-1, self.dims))(_input)
        
        # CNN
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
        # DNN
        flatten = layers.Flatten()(conv4)
        dense1=Dense(64, activation='relu')(flatten)
        dense1=Dropout(0.3)(dense1)

        # make this layer suitable for classification
        average_score=layers.GlobalAveragePooling1D(name='final')(dense1)
        
        model = Model(outputs=average_score, inputs=_input)
        
        return model


class FFN(object):
    
    def __init__(self, dims, n_targets):
        print('CNN init')
        self.dims = dims
        self.n_targets
        
    def build(self):
        _input = keras.Input(shape=(None, self.dims))

        # this layer might be redundant or unecessary
        re_input = layers.Reshape((-1, self.dims, 1), input_shape=(-1, self.dims))(_input)
        
        # Dense Layers
        d1 = Dense(64, activation='relu'))(re_input)
        d2 = Dense(64, activation='relu'))(d1)
        d3 = Dense(64, activation='relu'))(d2)
        d4 = Dense(64, activation='relu'))(d3)
        d5 = Dense(64, activation='relu'))(d4)
        
        dropout=Dropout(0.3)(d6)
        output = Dense(self.n_targets, activation='softmax', name='final')        
        model = Model(outputs=output, inputs=_input)
        
        return model
    
    

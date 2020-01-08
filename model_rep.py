import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
from keras import regularizers

class CNN(object):
    
    def __init__(self, dims, l2_val):
        print('CNN init')
        self.dims = dims
        self.l2_val
        
    def build(self):
        
        _input = keras.Input(shape=(self.dims))
        re_input = layers.BatchNormalization(input_shape=self.dims)(_input)
        
        # CNN
        conv1 = Conv1D(filters=16, kernel_size=3,input_shape=(self.dims),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(re_input)
        conv2 = Conv1D(filters=32, kernel_size=3,input_shape=(self.dims),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(conv1)
        conv3 = Conv1D(filters=64, kernel_size=3,input_shape=(self.dims),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(conv2)
        conv4 = Conv1D(filters=128, kernel_size=3,input_shape=(self.dims),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(conv3)

        # DNN
        flatten = layers.Flatten()(conv4)
        dense1=Dense(64, activation='relu')(flatten)
        dense1=Dropout(0.3)(dense1)

        average_score=layers.GlobalAveragePooling1D(name='avg')(dense1)
        
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
        d1 = Dense(64, activation='relu')(re_input)
        d2 = Dense(64, activation='relu')(d1)
        d3 = Dense(64, activation='relu')(d2)
        d4 = Dense(64, activation='relu')(d3)
        d5 = Dense(64, activation='relu')(d4)
        
        dropout=Dropout(0.3)(d6)

        # make this last layer output suitable for MSE and regression
        output = Dense(self.n_targets, name='avg')        
        model = Model(outputs=output, inputs=_input)
        
        return model
    
    

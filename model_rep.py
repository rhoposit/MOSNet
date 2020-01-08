import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers

class CNN(object):
    
    def __init__(self, dims, l2_val, dr):
        print('CNN init')
        self.dims = dims
        self.l2_val = l2_val
        self.dr = dr
        self.shape = (None, self.dims)
        
    def build(self):
        
        _input = keras.Input(shape=(self.shape))
        re_input = layers.BatchNormalization(input_shape=self.shape)(_input)
        
        # CNN
        conv1 = Conv1D(filters=16, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(re_input)
        conv2 = Conv1D(filters=32, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(conv1)
        conv3 = Conv1D(filters=64, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(conv2)
        conv4 = Conv1D(filters=128, kernel_size=3,activation='relu',kernel_regularizer=regularizers.l2(self.l2_val))(conv3)

        # DNN
#        flatten = layers.Flatten()(conv3)
#        dense1=Dense(64, activation='relu')(flatten)
#        dr=Dropout(self.dr)(dense1)
        dense2=Dense(1,activation='relu')(conv4)
        
        model = Model(outputs=dense2, inputs=_input)
        
        return model


class FFN(object):
    
    def __init__(self, dims, n_targets, dr):
        print('CNN init')
        self.dims = dims
        self.n_targets
        self.dr = dr
        self.shape = (None, self.dims)
        
    def build(self):
        _input = keras.Input(shape=self.shape)

        # Dense Layers
        d1 = Dense(64, activation='relu')(re_input)
        d2 = Dense(64, activation='relu')(d1)
        d3 = Dense(64, activation='relu')(d2)
        d4 = Dense(64, activation='relu')(d3)
        d5 = Dense(64, activation='relu')(d4)
        
        dropout=Dropout(self.dr)(d6)

        # make this last layer output suitable for MSE and regression
        output = Dense(self.n_targets, name='avg')        
        model = Model(outputs=output, inputs=_input)
        
        return model
    
    

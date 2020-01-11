import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers

class CNN(object):
    
    def __init__(self, dims, l2_val, dr):
        print('CNN init')
        self.dims = dims
        self.l2_val = l2_val
        self.dr = dr
        self.shape = (self.dims,1)
        
    def build(self, targets):
        k,m = 3,2
        model = Sequential()
        model.add(layers.BatchNormalization(input_shape=self.shape))
        model.add(Conv1D(filters=16, kernel_size=k,input_shape=(self.shape),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val)))
        model.add(MaxPooling1D(m))
        model.add(Conv1D(filters=32, kernel_size=k,input_shape=(self.shape),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val)))
        model.add(MaxPooling1D(m))
        model.add(Conv1D(filters=64, kernel_size=k,input_shape=(self.shape),activation='relu',kernel_regularizer=regularizers.l2(self.l2_val)))
        model.add(MaxPooling1D(m))
        model.add(layers.Flatten())

        if targets:
            model.add(Dense(10, activation='relu'))
        else:
            model.add(Dense(1, activation='softmax'))            
        vec = Input(shape=(self.shape))
        labels = model(vec)
        return Model(vec, labels)



class FFN(object):
    
    def __init__(self, dims, dr):
        print('FFN init')
        self.dims = dims
        self.dr = dr
        self.shape = (self.dims,1)
        
    def build(self, targets):
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(layers.BatchNormalization(input_shape=self.shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        
        if targets:
            model.add(Dense(10, activation='relu'))
        else:
            model.add(Dense(1, activation='softmax'))            
        vec = Input(shape=self.shape)
        labels = model(vec)
        return Model(vec, labels)
    
    

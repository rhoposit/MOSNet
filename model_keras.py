from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')
import numpy as np

import random
myseed = 1984
random.seed(myseed) 
np.random.seed(myseed)

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, TimeDistributed, Flatten, GlobalAveragePooling1D, LSTM, Bidirectional
from keras.layers.convolutional import Conv2D

class CNN(object):
    
    def __init__(self):
        print('CNN init')
        
    def build(self):
        _input = Input(shape=(None, 257))
        
        re_input = Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)
        
        # CNN
        conv1 = (Conv2D(8, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(8, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(8, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
        conv2 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
        conv3 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
        conv4 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
        # DNN
        flatten = TimeDistributed(Flatten())(conv4)
        dense1=TimeDistributed(Dense(32, activation='relu'))(flatten)
        dense1=Dropout(0.3)(dense1)

        frame_score=TimeDistributed(Dense(1), name='frame')(dense1)

        average_score=GlobalAveragePooling1D(name='avg')(frame_score)
        
        model = Model(outputs=[average_score, frame_score], inputs=_input)
        
        return model


class CNN_BLSTM(object):
    
    def __init__(self):
        print('CNN_BLSTM init')
        
    def build(self):
        _input = Input(shape=(None, 257))
        
        re_input = Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)
        
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
        
        re_shape = Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)

        # BLSTM
        blstm1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, 
                 recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), 
            merge_mode='concat')(re_shape)
    
        # DNN
        flatten = TimeDistributed(Flatten())(blstm1)
        dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
        dense1=Dropout(0.3)(dense1)

        frame_score=TimeDistributed(Dense(1), name='frame')(dense1)

        average_score=GlobalAveragePooling1D(name='avg')(frame_score)

        model = Model(outputs=[average_score, frame_score], inputs=_input)
        
        return model


class BLSTM(object):
    
    def __init__(self):
        print('BLSTM init')
        
    def build(self):
        _input = Input(shape=(None, 257))

        # BLSTM
        blstm1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, 
                 recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), 
            merge_mode='concat')(_input)
    
        # DNN
        flatten = TimeDistributed(Flatten())(blstm1)
        dense1=TimeDistributed(Dense(64, activation='relu'))(flatten)
        dense1=Dropout(0.3)(dense1)

        frame_score=TimeDistributed(Dense(1), name='frame')(dense1)

        average_score=GlobalAveragePooling1D(name='avg')(frame_score)
        
        model = Model(outputs=[average_score, frame_score], inputs=_input)
        
        return model

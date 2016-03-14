import numpy as np
import theano
import theano.tensor as T

import sys
sys.path.append('../../')

import LSTM.src.LSTMLayer

class LSTMNetwork(object):
    def __init__(self, number_of_layers=1):
        self.layers = {}

    def build_model(self):

        x = T.imatrix('x').astype(theano.config.floatX)
        drop_masks = T.imatrix('x').astype(theano.config.floatX)
        y = T.imatrix('y').astype(theano.config.floatX)

        self.layers[0] = LSTMLayer(input=x,drop_masks=drop_masks)
        for i in np.arange(1,self.number_of_layers):
            self.layers[i] = LSTMLayer()
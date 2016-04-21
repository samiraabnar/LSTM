import numpy as np
import theano
import theano.tensor as T

class FullyConnectedLayer(object):
    def __init__(self, random_state, input, input_dim, output_dim, W=None, bias=None, activation=T.tanh,layer_id="_0"):
        self.input = input
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_state = random_state
        self.activation = activation
        self.layer_id= layer_id


        self.W, self.bias = self.initialize_weights(W, bias)
        self.params = [self.W,self.bias]

        linear_output = T.dot(self.W,input) + self.bias
        if activation is None:
            self.output = linear_output
        else:
            self.output = activation(linear_output)[0]

    def initialize_weights(self, W, bias):
        if W is None:
            w_values = np.asarray(
                self.random_state.uniform(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim,self.input_dim))
                , dtype=theano.config.floatX)

            if self.activation == theano.tensor.nnet.sigmoid:
                w_values *= 4

            W = theano.shared(value=w_values, name="W"+self.layer_id, borrow="True")
        if bias is None:
            bias_values = np.zeros(self.output_dim, dtype=theano.config.floatX)
            bias = theano.shared(value=bias_values, name="bias"+self.layer_id, borrow="True")

        return W,bias



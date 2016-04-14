import numpy as np
import theano
import theano.tensor as T

import sys
sys.path.append('../../')

from LSTM.src.LSTMLayer import *
from LSTM.src.OutputLayer import *


from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *

class LSTMNetwork(object):
    def __init__(self, input_dim,output_dim,number_of_layers=1, hidden_dims=[100],dropout_p=0.5,learning_rate=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_layers = number_of_layers
        self.hidden_dims = hidden_dims
        self.random_state = np.random.RandomState(23455)
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p


        self.layers = {}
        self.build_model()


    def build_model(self):

        x = T.imatrix('x').astype(theano.config.floatX)
        drop_masks = T.imatrix('drop_masks').astype(theano.config.floatX)
        y = T.imatrix('y').astype(theano.config.floatX)

        self.layers[0] = LSTMLayer(random_state=self.random_state,input=x,drop_masks=drop_masks,input_dim=self.input_dim,output_dim=self.hidden_dims[0])
        params = self.layers[0].params
        for i in np.arange(1,self.number_of_layers):
            self.layers[i] = LSTMLayer(random_state=self.random_state,input=self.layers[i-1].output,drop_masks=np.ones_like(drop_masks,dtype=np.float32),input_dim=self.layers[i-1].output_dim,output_dim=self.hidden_dims[i])
            params += self.layers[i].params

        self.layers[self.number_of_layers] = OutputLayer(input=self.layers[self.number_of_layers - 1].output[-1],
                                                         input_dim = self.layers[self.number_of_layers -1].output_dim, output_dim= self.output_dim)


        cost = T.sum(T.nnet.categorical_crossentropy(self.layers[self.number_of_layers].probabilities, y[-1]))
        grads = T.grad(cost, params)

        updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(params,grads)]

        self.sgd_step = theano.function([x,drop_masks, y], [cost], updates=updates)

        self.test_model = theano.function([x,drop_masks, y], cost)

    def train_with_sgd(self, X_train, y_train, learning_rate=0.01, nepoch=1,
        callback_every=1, callback=None, *args):
        num_examples_seen = 0
        for epoch in range(nepoch):
            # For each training example...
            for i in np.random.permutation(len(y_train)):
                # One SGD step
                self.sgd_step(X_train[i],
                               [np.random.binomial(1, 1.0 - model.dropout_p,model.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                               ,y_train[i])
                num_examples_seen += 1
                # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(num_examples_seen, *args)

    def test_model(self,num_examples_seen):
        pc_sentiment = np.zeros((len(self.test["sentences"]),self.labels_count))
        for i in np.arange(len(self.test["sentences"])):
            pc_sentiment[i] = self.model.predict(self.test["sentences"][i],np.ones((len(self.test["sentences"][i]),self.model.hidden_dim),dtype=np.float32))

        correct = 0.0
        for i in np.arange(len(self.test["sentences"])):
            if np.argmax(pc_sentiment[i]) == np.argmax(self.test["sentiments"][i]):
                correct += 1

        accuracy = correct / len(self.test["sentences"])

        print("Accuracy on test: %f" %accuracy)


        pc_sentiment = np.zeros((len(self.train["sentences"]),self.labels_count))
        for i in np.arange(len(self.train["sentences"])):
            pc_sentiment[i] = self.model.predict(self.train["sentences"][i],np.ones((len(self.train["sentences"][i]),self.model.hidden_dim),dtype=np.float32))

        correct = 0.0
        for i in np.arange(len(self.train["sentences"])):
            if np.argmax(pc_sentiment[i]) == np.argmax(self.train["sentiments"][i]):
                correct += 1

        accuracy = correct / len(self.train["sentences"])

        print("Accuracy on train: %f" %accuracy)


if __name__ == '__main__':
    train = {}
    test = {}
    train["sentences"], train["sentiments"], word_to_index, index_to_word, labels_count = DataPrep.load_one_hot_sentiment_data("../data/sentiment/trainsentence_and_label_binary_words_added.txt")
    test["sentences"], test["sentiments"]= DataPrep.load_one_hot_sentiment_data_traind_vocabulary("../data/sentiment/testsentence_and_label_binary.txt",word_to_index, index_to_word,labels_count)
    model = LSTMNetwork(input_dim=len(index_to_word), output_dim=labels_count)

    expected_outputs = []
    for i in np.arange(len(train["sentences"])):
        s_out = np.zeros((len(train["sentences"][i]),labels_count),dtype=np.float32)
        s_out[-1] = train["sentiments"][i]
        expected_outputs.append(s_out)
    print("training ... ")
    model.train_with_sgd(train["sentences"],expected_outputs,learning_rate=0.01, nepoch=1,
        callback_every=1, callback=model.test_model)


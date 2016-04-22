import numpy as np
import pickle
import theano
import theano.tensor as T

import sys
sys.path.append('../../')

from LSTM.src.LSTMLayer import *
from LSTM.src.FullyConnectedLayer import *
from LSTM.src.OutputLayer import *
from LSTM.src.WordEmbeddingLayer import *


from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *
from Util.util.nnet.LearningAlgorithms import *

class FullyConnectedLSTM(object):
    def __init__(self, input_dim,output_dim,number_of_layers=1, hidden_dims=[100],dropout_p=0.5,learning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_layers = number_of_layers
        self.hidden_dims = hidden_dims
        self.random_state = np.random.RandomState(23455)
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p


        self.layers = {}


    def build_model(self):
        x = T.matrix('x').astype(theano.config.floatX)
        y = T.ivector('y')

        params = []
        self.layers[0] = LSTMLayer(input=x,
                                             input_dim=self.input_dim,
                                             output_dim=self.output_dim,
                                             random_state=self.random_state,layer_id="_0")
        params += self.layers[0].params

        """self.layers[1] = FullyConnectedLayer(input=T.mean(self.layers[0].output,axis=0),
                                             input_dim=self.layers[0].output_dim,
                                             output_dim=self.output_dim,
                                             random_state=self.random_state,activation=T.nnet.softmax,layer_id="_1")
        params += self.layers[1].params"""

        L1 = 0.001 * T.sum([T.sum(param) for param in params])
        L2 = 0.001 * T.sum([T.sum(param ** 2) for param in params])
        cost = T.sum(T.nnet.binary_crossentropy(self.layers[0].output[-1],y)) + L1 + L2

        #grads = T.grad(cost, params)

        #updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(params,grads)]
        updates =  LearningAlgorithms.adam(cost,params,lr=0.005)

        self.sgd_step = theano.function([x,y],cost, updates=updates)
        self.predict = theano.function([x],self.layers[0].output[-1])

        self.test_model = theano.function([x, y], cost)


    def train(self, X_train, y_train,X_dev,y_dev,nepoch=100):
        for epoch in range(nepoch):
            # For each training example...
            for i in np.random.permutation(len(y_train)):
                # One SGD step
                y_train[i]
                cost = self.sgd_step(np.asarray(X_train[i], dtype=np.float32)
                               #,[np.random.binomial(1, 1.0 - self.dropout_p,self.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                               ,y_train[i].astype(np.int32))
                #print(cost)
                #exit(0)

            print("Accuracy on dev: ")
            self.test_dev(X_dev,y_dev)
            print("Accuracy on train: ")
            self.test_dev(X_train,y_train)

    def test_dev(self,X_dev,y_dev):
        pc_sentiment = np.zeros(len(X_dev))
        for i in np.arange(len(X_dev)):
            pc_sentiment[i] = np.argmax(self.predict(np.asarray(X_dev[i],dtype=np.float32)
                                                     #,np.ones((len(X_dev[i]),self.input_dim),dtype=np.float32)
                                                     ))

        correct = 0.0
        for i in np.arange(len(X_dev)):
            if pc_sentiment[i] == np.argmax(y_dev[i]):
                correct += 1

        accuracy = correct / len(X_dev)

        print(accuracy)


    @staticmethod
    def train_1layer_glove_wordembedding(hidden_dim,modelfile):
        train = {}
        test = {}
        dev = {}

        embedded_train, train_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="train",representation="glove.840B.300d")
        embedded_dev, dev_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="dev",representation="glove.840B.300d")
        embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="test",representation="glove.840B.300d")

        flstm = FullyConnectedLSTM(input_dim=len(embedded_train[0][0]),output_dim=len(train_labels[0]),number_of_layers=1, hidden_dims=[hidden_dim],dropout_p=0.5,learning_rate=0.01)
        flstm.build_model()

        train_labels[train_labels == 0] = -1
        dev_labels[dev_labels == 0] = -1
        flstm.train(embedded_train,train_labels,embedded_dev,dev_labels)
        flstm.save_model(modelfile)

    def save_model(self,modelfile):
        with open(modelfile,"wb") as f:
            pickle.dump(self.layers,f)

    @staticmethod
    def load_model(modelfile):
        with open(modelfile,"rb") as f:
            return pickle.load(f)




if __name__ == '__main__':
    FullyConnectedLSTM.train_1layer_glove_wordembedding(100,"f_model.txt")

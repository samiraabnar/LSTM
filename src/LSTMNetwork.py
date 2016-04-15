import numpy as np
import theano
import theano.tensor as T

import sys
sys.path.append('../../')

from LSTM.src.LSTMLayer import *
from LSTM.src.OutputLayer import *
from LSTM.src.WordEmbeddingLayer import *


from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *
from WordEmbeddingLayer import *

class LSTMNetwork(object):
    def __init__(self, input_dim,output_dim,number_of_layers=1, hidden_dims=[100],dropout_p=0.0,learning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_layers = number_of_layers
        self.hidden_dims = hidden_dims
        self.random_state = np.random.RandomState(23455)
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p


        self.layers = {}


    def build_model_1(self):

        x = T.imatrix('x').astype(theano.config.floatX)
        drop_masks = T.imatrix('drop_masks').astype(theano.config.floatX)
        y = T.imatrix('y').astype(theano.config.floatX)

        self.layers[0] = LSTMLayer(random_state=self.random_state,input=x,drop_masks=drop_masks,input_dim=self.input_dim,output_dim=self.hidden_dims[0])
        params = self.layers[0].params
        for i in np.arange(1,self.number_of_layers):
            self.layers[i] = LSTMLayer(random_state=self.random_state,input=self.layers[i-1].output,drop_masks=np.ones_like(drop_masks,dtype=np.float32),input_dim=self.layers[i-1].output_dim,output_dim=self.hidden_dims[i])
            params += self.layers[i].params

        self.layers[self.number_of_layers] = OutputLayer(input=self.layers[self.number_of_layers - 1].output,
                                                         input_dim=self.layers[self.number_of_layers -1].output_dim, output_dim=self.output_dim)

        params += self.layers[self.number_of_layers].params

        cost = T.sum(T.nnet.categorical_crossentropy(self.layers[self.number_of_layers].probabilities[-1], y[-1]))
        grads = T.grad(cost, params)

        updates =  [(param_i, param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(params,grads)]
        #updates =  self.adam(cost,params)

        self.sgd_step = theano.function([x,drop_masks, y], [cost], updates=updates)
        self.predict = theano.function([x,drop_masks],self.layers[self.number_of_layers].predictions)

        self.test_model = theano.function([x,drop_masks, y], cost)


    def adam(self,loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
         epsilon=1e-8, gamma=1-1e-7):
        """
        ADAM update rules
        Default values are taken from [Kingma2014]
        References:
        [Kingma2014] Kingma, Diederik, and Jimmy Ba.
        "Adam: A Method for Stochastic Optimization."
        arXiv preprint arXiv:1412.6980 (2014).
        :parameters:
            - loss : Theano expression
                specifying loss
            - all_params : list of theano.tensors
                Gradients are calculated w.r.t. tensors in all_parameters
            - learning_Rate : float
            - beta1 : float
                Exponentioal decay rate on 1. moment of gradients
            - beta2 : float
                Exponentioal decay rate on 2. moment of gradients
            - epsilon : float
                For numerical stability
            - gamma: float
                Decay on first moment running average coefficient
            - Returns: list of update rules
        """

        updates = []
        all_grads = theano.grad(loss, all_params)

        i = theano.shared(np.float32(1))  # HOW to init scalar shared?
        i_t = i + 1.
        fix1 = 1. - (1. - beta1)**i_t
        fix2 = 1. - (1. - beta2)**i_t
        beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED
        learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)

        for param_i, g in zip(all_params, all_grads):
            m = theano.shared(
                np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
            v = theano.shared(
                np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))

            m_t = (beta1_t * g) + ((1. - beta1_t) * m) # CHANGED from b_t TO use beta1_t
            v_t = (beta2 * g**2) + ((1. - beta2) * v)
            g_t = m_t / (T.sqrt(v_t) + epsilon)
            param_i_t = param_i - (learning_rate_t * g_t)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param_i, param_i_t) )
        updates.append((i, i_t))

        return updates

    def train(self, X_train, y_train,X_dev,y_dev,nepoch=100):
        for epoch in range(nepoch):
            # For each training example...
            for i in np.random.permutation(len(y_train)):
                # One SGD step
                s_out = np.zeros((len(X_train[i]),self.output_dim),dtype=np.float32)
                s_out[-1] = y_train[i]
                self.sgd_step(np.asarray(X_train[i], dtype=np.float32),
                               [np.random.binomial(1, 1.0 - self.dropout_p,self.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                               ,s_out)

            self.test_dev(X_train,y_train)

    def test_dev(self,X_dev,y_dev):
        pc_sentiment = np.zeros(len(X_dev))
        for i in np.arange(len(X_dev)):
            pc_sentiment[i] = np.argmax(self.predict(np.asarray(X_dev[i],dtype=np.float32),np.ones((len(X_dev[i]),self.input_dim),dtype=np.float32)))

        correct = 0.0
        for i in np.arange(len(X_dev)):
            if pc_sentiment[i] == np.argmax(y_dev[i]):
                correct += 1

        accuracy = correct / len(X_dev)

        print("Accuracy on dev: %f" %accuracy)


    @staticmethod
    def train_1layaer_glove_wordembedding():
        train = {}
        test = {}
        dev = {}
        index_to_word=[UNKNOWN_TOKEN]
        """train["sentences"], train["sentiments"], word_to_index, index_to_word, labels_count = DataPrep.load_sentiment_data("../data/sentiment/trainsentence_and_label_binary.txt",index_to_word)
        test["sentences"], test["sentiments"] , word_to_index, index_to_word, lc= DataPrep.load_sentiment_data("../data/sentiment/testsentence_and_label_binary.txt",index_to_word,labels_count)
        dev["sentences"], dev["sentiments"] , word_to_index, index_to_word, lc= DataPrep.load_sentiment_data("../data/sentiment/devsentence_and_label_binary.txt",index_to_word,labels_count)

        vocab_representation = WordEmbeddingLayer()
        vocab_representation.load_embeddings_from_glove_file(filename="../data/glove.840B.300d.txt",filter=index_to_word)
        vocab_representation.save_embedding("../data/filtered_glove.840B.300d")
        vocab_representation.load_filtered_embedding("../data/filtered_glove.840B.300d")
        vocab_representation.embed_and_save(sentences=train["sentences"],labels=train["sentiments"],path="../data/",name="train",representation="glove.840B.300d")
        vocab_representation.embed_and_save(sentences=dev["sentences"],labels=dev["sentiments"],path="../data/",name="dev",representation="glove.840B.300d")
        vocab_representation.embed_and_save(sentences=test["sentences"],labels=test["sentiments"],path="../data/",name="test",representation="glove.840B.300d")"""

        embedded_train, train_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="train",representation="glove.840B.300d")
        embedded_dev, dev_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="dev",representation="glove.840B.300d")
        embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="test",representation="glove.840B.300d")

        lstm = LSTMNetwork(input_dim=len(embedded_train[0][0]),output_dim=len(train_labels[0]),number_of_layers=1, hidden_dims=[100],dropout_p=0.5,learning_rate=0.01)
        lstm.build_model_1()

        lstm.train(embedded_train,train_labels,embedded_dev,dev_labels)





if __name__ == '__main__':
    """train = {}
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

    print(cost)"""
    LSTMNetwork.train_1layaer_glove_wordembedding()

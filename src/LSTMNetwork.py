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
from Util.util.nnet.LearningAlgorithms import *


class LSTMNetwork(object):
    def __init__(self, input_dim,output_dim,number_of_layers=1, hidden_dims=[100],dropout_p=0.5,learning_rate=0.1):
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
        y = T.ivector('y')

        self.layers[0] = LSTMLayer(random_state=self.random_state,input=x,drop_masks=drop_masks,input_dim=self.input_dim,output_dim=self.hidden_dims[0])
        params = self.layers[0].params

        self.layers[1] = OutputLayer(input=self.layers[0].output,
                                                         input_dim=self.layers[0].output_dim, output_dim=self.output_dim,random_state=self.random_state)

        params += self.layers[1].params
        _EPSILON = 10e-8

        L1 = 0.001 * T.sum([T.sum(param) for param in params])
        L2 = 0.001 * T.sum([T.sum(param ** param) for param in params])
        cost = T.sum(T.nnet.categorical_crossentropy(T.clip(self.layers[self.number_of_layers].probabilities[-1], _EPSILON, 1.0 - _EPSILON),y)) + L1 + L2

        #grads = T.grad(cost, params)

        #updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(params,grads)]
        updates =  LearningAlgorithms.adam(cost,params,learning_rate=0.001)

        self.sgd_step = theano.function([x,drop_masks, y], L1, updates=updates)
        self.predict = theano.function([x,drop_masks],self.layers[self.number_of_layers].probabilities[-1])

        self.test_model = theano.function([x,drop_masks, y], cost)





    def train(self, X_train, y_train,X_dev,y_dev,nepoch=100):
        for epoch in range(nepoch):
            # For each training example...
            for i in np.random.permutation(len(y_train)):
                # One SGD step
                y_train[i]
                cost = self.sgd_step(np.asarray(X_train[i], dtype=np.float32),
                               [np.random.binomial(1, 1.0 - self.dropout_p,self.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                               ,y_train[i].astype(np.int32))
                print(cost)

            print("Accuracy on dev: ")
            self.test_dev(X_dev,y_dev)
            print("Accuracy on train: ")
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

        print(accuracy)


    @staticmethod
    def train_1layer_glove_wordembedding(hidden_dim,modelfile):
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

        lstm = LSTMNetwork(input_dim=len(embedded_train[0][0]),output_dim=len(train_labels[0]),number_of_layers=1, hidden_dims=[hidden_dim],dropout_p=0.5,learning_rate=0.01)
        lstm.build_model_1()

        lstm.train(embedded_train,train_labels,embedded_dev,dev_labels)
        lstm.save_model(modelfile)

    def save_model(self,modelfile):
        with open(modelfile) as f:
            f.dump(self,f)

    @staticmethod
    def load_model(modelfile):
        with open(modelfile) as f:
            return f.load(f)







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

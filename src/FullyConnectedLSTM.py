import numpy as np
import pickle
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


import sys
sys.path.append('../../')

from LSTM.src.LSTMLayer import *
from LSTM.src.FullyConnectedLayer import *
from LSTM.src.OutputLayer import *
from LSTM.src.WordEmbeddingLayer import *


from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *
from Util.util.nnet.LearningAlgorithms import *

from six.moves import cPickle

class FullyConnectedLSTM(object):
    def __init__(self, input_dim,output_dim,number_of_layers=1, hidden_dims=[100],dropout_p=0.1,learning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_layers = number_of_layers
        self.hidden_dims = hidden_dims
        self.random_state = np.random.RandomState(23455)
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p


        self.layers = {}

    def build_loaded_model(self,layers):
        x = T.matrix('x').astype(theano.config.floatX)
        self.layers = layers
        self.predict = theano.function([self.layers[0].input],self.layers[self.number_of_layers - 1].output[-1])

        U_i,U_f,U_o,U,W_i,W_f,W_o,W,b_i,b_f,b_o,b = self.layers[0].U_input, self.layers[0].U_forget, self.layers[0].U_output,self.layers[0].U,\
                                                    self.layers[0].W_input, self.layers[0].W_forget, self.layers[0].W_output,self.layers[0].W, \
                                                    self.layers[0].bias_input, self.layers[0].bias_forget, self.layers[0].bias_output \
                                                    ,self.layers[0].bias
        def forward_step(x_t, prev_state,prev_content):
            input_gate = T.nnet.hard_sigmoid(T.dot((U_i),x_t) + T.dot(W_i,prev_state) + b_i)
            forget_gate = T.nnet.hard_sigmoid(T.dot((U_f),x_t) + T.dot(W_f,prev_state)+ b_f)
            output_gate = T.nnet.hard_sigmoid(T.dot((U_o),x_t) + T.dot(W_o,prev_state)+ b_o)



            stabilized_input = T.tanh(T.dot((U),x_t) + T.dot(W,prev_state) + b)

            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)
            o = T.nnet.softmax(s)[0]

            return [o,s,c,input_gate,forget_gate,output_gate]

        [self.output,hidden_state,memory_content,self.input_gate,self.forget_gate,self.output_gate] , updates = theano.scan(
            forward_step,
            sequences=[x],
            truncate_gradient=-1,
            outputs_info=[None,dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX))
                          ,None,None,None
                          ])


        self.get_gates =theano.function([x],[self.output,self.input_gate,self.forget_gate,self.output_gate])



    def build_model(self):
        x = T.matrix('x').astype(theano.config.floatX)
        y = T.ivector('y')

        params = []
        self.layers[0] = LSTMLayer(input=x,
                                             input_dim=self.input_dim,
                                             output_dim=self.hidden_dims[0],
                                             outer_output_dim=self.output_dim,
                                             random_state=self.random_state,layer_id="_0")
        params += self.layers[0].params

        """self.layers[1] = FullyConnectedLayer(input=self.layers[0].output[-1],
                                             input_dim=self.layers[0].output_dim,
                                             output_dim=self.output_dim,
                                             random_state=self.random_state,activation=T.nnet.softmax,layer_id="_1")
        params += self.layers[1].params"""


        L1 = 0.0005 * T.sum([T.sum(param) for param in params])
        L2 = 0.0005 * T.sum([T.sum(param ** 2) for param in params])
        off = 1e-8

        cost = -T.log(self.layers[0].output[-1][T.argmax(y)] + off) + L1 + L2

        #cost = T.sum(T.nnet.binary_crossentropy(self.layers[0].output[-1],y)) + L1 + L2

        grads = T.grad(theano.gradient.grad_clip(cost,-1,1), params)
        self.learning_rate = 0.005
        #updates = [(param_i, param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(params,grads)]
        updates =  LearningAlgorithms.adam(cost,params,lr=0.000001)

        self.sgd_step = theano.function([x,y],cost, updates=updates)
        self.predict = theano.function([x],self.layers[0].output[-1])

        self.test_model = theano.function([x, y], cost)


    def train(self, X_train, y_train,X_dev,y_dev,nepoch=100):
        for epoch in range(nepoch):
            # For each training example...
            for i in np.random.permutation(len(y_train)):
                # One SGD step
                y_train[i]
                cost = self.sgd_step(np.asarray(X_train[i], dtype=np.float32) * [np.random.binomial(1, 1.0 - self.dropout_p,self.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                               #,[np.random.binomial(1, 1.0 - self.dropout_p,self.input_dim).astype(dtype=np.float32) for i in np.arange(len(X_train[i]))]
                               ,y_train[i].astype(np.int32))
                print(cost)
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
        #embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="test",representation="glove.840B.300d")


        flstm = FullyConnectedLSTM(input_dim=len(embedded_train[0][0]),output_dim=len(train_labels[0]),number_of_layers=1, hidden_dims=[hidden_dim],dropout_p=0.1,learning_rate=0.01)
        flstm.build_model()

        #train_labels[train_labels == 0] = -1
        #dev_labels[dev_labels == 0] = -1
        flstm.train(embedded_train,train_labels,embedded_dev,dev_labels)
        flstm.save_model(modelfile)

    def save_model(self,modelfile):
        with open(modelfile,"wb") as f:
            cPickle.dump(self.layers,f,protocol=cPickle.HIGHEST_PROTOCOL)

        with open("params_"+modelfile,"wb") as f:
            for layer_key in self.layers.keys():
                cPickle.dump(self.layers[layer_key].params,f,protocol=cPickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_model(modelfile):
        layers = {}
        with open(modelfile,"rb") as f:
            layers = cPickle.load(f)
        with open("params_"+modelfile,"rb") as f:
            for layer_key in layers.keys():
               layers[layer_key].params = cPickle.load(f)

        n_of_layers=len(layers.keys())

        flstm = FullyConnectedLSTM(input_dim=layers[0].input_dim,output_dim=layers[n_of_layers-1].output_dim,number_of_layers=n_of_layers, hidden_dims=[layers[0].output_dim])
        flstm.build_loaded_model(layers)
        embedded_test, test_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="test",representation="glove.840B.300d")
        print("Accuracy on test: ")
        flstm.test_dev(embedded_test[0:10],test_labels[0:10])
        print("Accuracy on train: ")
        embedded_train, train_labels = WordEmbeddingLayer.load_embedded_data(path="../data/",name="train",representation="glove.840B.300d")
        flstm.test_dev(embedded_train[0:10], train_labels[0:10])
        return flstm


    @staticmethod
    def lstm_forward_pass(embeddedSent,flstm):
        outputs,i_gates, f_gates,o_gates =  flstm.forward(embeddedSent)




    @staticmethod
    def show_sentiment_path(sentence,vocab_representation,flstm):
        tokens = sentence.split()
        embedded = vocab_representation.embed([tokens])[0]

        predictions = []
        labels = []
        gates = []
        for i in np.arange(0,len(embedded)):
            labels.append(tokens[i])
            predictions.append(flstm.predict(np.asarray(embedded[0:i+1],dtype=np.float32)).tolist())
            gates = flstm.get_gates(np.asarray(embedded[0:i+1],dtype=np.float32))

        vis_data = predictions

        fig = plt.figure()
        ax = {}
        ax[0] = fig.add_subplot(111, projection='3d')

        fig2 = plt.figure()

        for i in np.arange(0,len(gates)):
            for k in np.arange(0,len(embedded)):
                ax[1+(i*len(tokens))+k] = fig2.add_subplot(4,len(tokens),(i*len(tokens))+k+1)
                ax[1+(i*len(tokens))+k].bar(np.arange(len(gates[i][k])),gates[i][k],0.1)
                ax[1+(i*len(tokens))+k].set_ylim(0,1.0)
                if(i == 0):
                    ax[1+(i*len(tokens))+k].set_title(tokens[k])



        vis_x = [x[0] for x in vis_data]
        vis_y = [x[1] for x in vis_data]
        vis_z = [x[2] for x in vis_data]

        ax[0].plot_wireframe(vis_x, vis_y,vis_z, linestyle='-')
        ax[0].scatter(vis_x, vis_y,vis_z,marker='o',depthshade=True)
        for label, x,y,z in zip(labels, vis_x,vis_y,vis_z):
            ax[0].text(x,y,z,label)

        ax[0].set_xlim3d(0, 1)
        ax[0].set_ylim3d(0, 1)
        ax[0].set_zlim3d(0, 1)

        ax[0].set_xlabel('Negative')
        ax[0].set_ylabel('Neutral')
        ax[0].set_zlabel('Positive')
        plt.show()






    @staticmethod
    def analyse():
        flstm = FullyConnectedLSTM.load_model("test_model.txt")

        vocab_representation = WordEmbeddingLayer()
        vocab_representation.load_filtered_embedding("../data/filtered_glove.840B.300d")

        FullyConnectedLSTM.show_sentiment_path("he is a bad actor , but his play is good !",vocab_representation,flstm)
        embed_sent = vocab_representation.embed(sentences=[["bad","!"]])[0]
        print("bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["not","bad","!"]])[0]
        print("not bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["very","bad","!"]])[0]
        print("very bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["it","is","not","good","!"]])[0]
        print("it is not good! is: "+str(np.argmax(flstm.predict(np.asarray(embed_sent,dtype=np.float32)))))

        embed_sent = vocab_representation.embed(sentences=[["it","is","not","good","not","bad","!"]])[0]
        print("it is not good not bad! is: "+str(np.argmax(flstm.predict(np.asarray(embed_sent,dtype=np.float32)))))

        embed_sent = vocab_representation.embed(sentences=[["good","or","bad","!"]])[0]
        print("good or bad! is: "+str(np.argmax(flstm.predict(np.asarray(embed_sent,dtype=np.float32)))))

        embed_sent = vocab_representation.embed(sentences=[["good","or","bad","!"]])[0]
        print("bad or good! is: "+str(np.argmax(flstm.predict(np.asarray(embed_sent,dtype=np.float32)))))


        embed_sent = vocab_representation.embed(sentences=[["the"]])[0]
        print("the is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["bad"]])[0]
        print("bad is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["movie"]])[0]
        print("movie is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["is"]])[0]
        print("is is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["made"]])[0]
        print("made is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["by"]])[0]
        print("by is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["me"]])[0]
        print("me is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","bad","movie","is","made","by","me","."]])[0]
        print("the bad movie is made by me. is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","bad","movie","is","made","by","a","good","man","."]])[0]
        print("the bad movie is made by a good man. is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","movie","is","made","by","a","good","man","."]])[0]
        print("the movie is made  by a good man. is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","movie","made","by","a","good","man","is","bad","."]])[0]
        print("the movie made  by a good man is bad. is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","movie","made","by","a","bad","man","is","good","."]])[0]
        print("the movie made  by a bad man is good. is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","bad","man","made","a","good","movie","."]])[0]
        print("the bad man made  a good movie. is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","good","man","made","a","bad","movie","."]])[0]
        print("the good man made  a bad movie. is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","good","man","is","bad","!"]])[0]
        print("the good man is bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","actor","is","bad","!"]])[0]
        print("the actor is bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","actor","played","bad","!"]])[0]
        print("the actor played bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","good","actor","played","bad","!"]])[0]
        print("the good actor played so bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["I","thought","it","should","be","bad","but","it","was","good","!"]])[0]
        print("I thought it should be bad but it was good ! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["I","thought","it","should","be","bad",",","but","it","was","good","!"]])[0]
        print("I thought it should be bad, but it was good ! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","actor","is","normally","bad",",","but","he","played","good","!"]])[0]
        print("the actor is normally bad, but he played good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","actor","is","bad",",","but","he","played","good","!"]])[0]
        print("the actor is bad, but he played good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","actor","is","normally","bad","!"]])[0]
        print("the actor is normally bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["the","actor","is","bad","!"]])[0]
        print("the actor is bad! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["he","played","good","!"]])[0]
        print("he played good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["his","play","is","good","!"]])[0]
        print("his play is good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["he","is","a","bad","actor","he","played","good","!"]])[0]
        print("he is a bad actor, he played good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["he","is","a","bad","actor",",","but","he","played","good","!"]])[0]
        print("he is a bad actor, but he played good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["he","is","a","bad","actor",",","but","his","play","is","good","!"]])[0]
        print("he is a bad actor, but his play is good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["he","is","a","bad","actor",",","but","his","play","is","good","!"]])[0]
        print("he is a bad actor, but his play is good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))


        embed_sent = vocab_representation.embed(sentences=[["although", "he","is","a","bad","actor",",","he","played","good","!"]])[0]
        print("although he is a bad actor, he played good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["although", "he","is","a","bad","actor",",","his","play","is","good","!"]])[0]
        print("although he is a bad actor, his play is good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))


        embed_sent = vocab_representation.embed(sentences=[["although", "he","is","a","bad","actor",",","his","act","is","good","!"]])[0]
        print("although he is a bad actor, his act is good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["although", "he","is","a","bad","actor",",","his","play","is","good","!"]])[0]
        print("although he is a bad actor, his play is good! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))

        embed_sent = vocab_representation.embed(sentences=[["they", "made","a","bad","movie","from","a","good","story","!"]])[0]
        print("they made a bad movie from a good story! is: "+str(flstm.predict(np.asarray(embed_sent,dtype=np.float32))))




if __name__ == '__main__':
    FullyConnectedLSTM.train_1layer_glove_wordembedding(50,"test_model_diffdim.txt")
    #FullyConnectedLSTM.load_model("test_model.txt")
    #FullyConnectedLSTM.analyse()


import numpy as np

import theano
import theano.tensor as T

class OutputLayer(object): #Y = softmax( Wx + bias)

    def __init__(self, input, input_dim, output_dim,random_state):
        self.input = input
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_state=random_state
        self.initialize_params()

        self.build_model()

    def initialize_params(self):
        w_value = np.asarray(
            self.random_state.uniform(-np.sqrt(1.0 / (self.output_dim + self.input_dim)),
                                      np.sqrt(1.0 / (self.output_dim + self.input_dim)),
                                      (self.output_dim,self.input_dim))
            , dtype=theano.config.floatX)
        self.W = theano.shared(value=w_value, name="W", borrow=True)
        bias_value = np.zeros(self.output_dim, dtype=theano.config.floatX)
        self.bias = theano.shared(value=bias_value, name="bias", borrow=True)

        self.params = [self.W, self.bias]

    def negative_log_likelihood(self,train_y):
        return -T.mean(T.log(self.probabilities)[T.arange(train_y.shape[0]), train_y])

    def errors(self,y):
        return T.mean(T.neq(self.predictions, y))

    def build_model(self):

         def forward_step(x_t):
            o = T.nnet.softmax(T.dot(self.W,x_t)+ self.bias)[0]

            return [o]

         self.probabilities, updates = theano.scan(
            forward_step,
            sequences=[self.input],
            outputs_info=[None])

         self.predictions = T.eye(self.output_dim)[T.argmax(self.probabilities,axis=1)]

        #cost = self.negative_log_likelihood(y)
        #all_params = [self.W, self.bias]
        #grads = T.grad(cost, all_params)

        #updates = [ (param_i, param_i - self.learning_rate*grad_i) for (param_i,grad_i) in zip(all_params,grads)]
        # train = theano.function([x,y], cost, updates=updates)







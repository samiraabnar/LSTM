import theano
import theano.tensor as T
import numpy as np


class LSTMLayer(object):

    def __init__(self,random_state,input,drop_masks,input_dim,output_dim,bptt_truncate=-1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_state = random_state
        self.initialize_params()
        self.bptt_truncate = bptt_truncate

        def forward_step(x_t,dropmask_t, prev_state,prev_content):
            input_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_input),x_t) + T.dot(self.W_input,prev_state) + self.bias_input)
            forget_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_forget),x_t) + T.dot(self.W_forget,prev_state)+ self.bias_forget)
            output_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_output),x_t) + T.dot(self.W_output,prev_state)+ self.bias_output)

            stabilized_input = T.tanh(T.dot((dropmask_t * self.U),x_t) + T.dot(self.W,prev_state) + self.bias)

            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)

            return [s,c]

        [hidden_state,memory_content] = theano.scan(
            forward_step,
            sequences=[input,drop_masks],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.output_dim)),
                          dict(initial=T.zeros(self.output_dim))])

        self.output = hidden_state[-1]

    def initialize_params(self):
        U_input = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      (self.output_dim,self.input_dim))
            , dtype=theano.config.floatX)

        U_forget = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      (self.output_dim,self.input_dim))
            , dtype=theano.config.floatX)

        U_output = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      (self.output_dim,self.input_dim))
            , dtype=theano.config.floatX)


        W_input = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      (self.output_dim,self.output_dim))
            , dtype=theano.config.floatX)

        W_forget = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      (self.output_dim,self.output_dim))
            , dtype=theano.config.floatX)

        W_output = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      (self.output_dim,self.output_dim))
            , dtype=theano.config.floatX)


        U = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                      (self.output_dim,self.input_dim))
            , dtype=theano.config.floatX)

        W = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      (self.output_dim,self.output_dim))
            , dtype=theano.config.floatX)

        bias_input = np.zeros(self.output_dim, dtype=theano.config.floatX)
        bias_forget = np.zeros(self.output_dim, dtype=theano.config.floatX)
        bias_output = np.zeros(self.output_dim, dtype=theano.config.floatX)
        bias = np.zeros(self.output_dim, dtype=theano.config.floatX)

        self.W = theano.shared(value=W, name="W", borrow="True")
        self.U = theano.shared(value=U, name="U", borrow="True")
        self.bias = theano.shared(value=bias, name="bias", borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input", borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input", borrow="True")
        self.bias_input = theano.shared(value=bias_input, name="bias_input", borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output", borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output", borrow="True")
        self.bias_output = theano.shared(value=bias_output, name="bias_output", borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget", borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget", borrow="True")
        self.bias_forget = theano.shared(value=bias_forget, name="bias_forget", borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output, self.bias_input, self.bias_forget, self.bias_output, self.U, self.W, self.bias]
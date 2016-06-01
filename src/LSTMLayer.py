import theano
import theano.tensor as T
import numpy as np


class LSTMLayer(object):

    def __init__(self,random_state,input,input_dim,output_dim,outer_output_dim,bptt_truncate=-1,layer_id="_0"):
        self.input = input
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.outer_output_dim = outer_output_dim
        self.random_state = random_state
        self.layer_id = layer_id
        self.initialize_params()
        self.bptt_truncate = bptt_truncate

        """
         def forward_step(x_t,dropmask_t, prev_state,prev_content):
            input_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_input),x_t) + T.dot(self.W_input,prev_state) + self.bias_input)
            forget_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_forget),x_t) + T.dot(self.W_forget,prev_state)+ self.bias_forget)
            output_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_output),x_t) + T.dot(self.W_output,prev_state)+ self.bias_output)

            stabilized_input = T.nnet.softmax(T.dot((dropmask_t * self.U),x_t) + T.dot(self.W,prev_state) + self.bias)[0]

            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)
            o = T.nnet.logsoftmax(s)[0]

            return [o,s,c]
        """



        def forward_step(x_t, prev_state,prev_content):
            input_gate = T.nnet.hard_sigmoid(T.dot(( self.U_input),x_t) + T.dot(self.W_input,prev_state) + self.bias_input)
            forget_gate = T.nnet.hard_sigmoid(T.dot(( self.U_forget),x_t) + T.dot(self.W_forget,prev_state)+ self.bias_forget)
            output_gate = T.nnet.hard_sigmoid(T.dot((self.U_output),x_t) + T.dot(self.W_output,prev_state)+ self.bias_output)



            stabilized_input = T.tanh(T.dot((self.U),x_t) + T.dot(self.W,prev_state) + self.bias)

            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)
            o = T.nnet.sigmoid(T.dot(self.O_w,s)+self.O_bias)

            return [o,s,c]

        [self.output,self.hidden_state,self.memory_content] , updates = theano.scan(
            forward_step,
            sequences=[self.input],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX))
                          ])

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

        self.W = theano.shared(value=W, name="W"+self.layer_id, borrow="True")
        self.U = theano.shared(value=U, name="U"+self.layer_id, borrow="True")
        self.bias = theano.shared(value=bias, name="bias", borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input"+self.layer_id, borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input"+self.layer_id, borrow="True")
        self.bias_input = theano.shared(value=bias_input, name="bias_input"+self.layer_id, borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output"+self.layer_id, borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output"+self.layer_id, borrow="True")
        self.bias_output = theano.shared(value=bias_output, name="bias_output"+self.layer_id, borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget"+self.layer_id, borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget"+self.layer_id, borrow="True")
        self.bias_forget = theano.shared(value=bias_forget, name="bias_forget"+self.layer_id, borrow="True")


        O_w = np.asarray(
            self.random_state.uniform(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      (self.outer_output_dim,self.output_dim))
            , dtype=theano.config.floatX)

        O_bias = np.zeros(self.outer_output_dim, dtype=theano.config.floatX)

        self.O_w = theano.shared(value=O_w, name="O_w"+self.layer_id, borrow="True")
        self.O_bias = theano.shared(value=O_bias, name="O_bias"+self.layer_id, borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.bias_input, self.bias_forget, self.bias_output, self.U, self.W, self.bias]

        self.output_params = [self.O_w,self.O_bias]
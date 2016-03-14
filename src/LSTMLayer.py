import theano
import theano.tensor as T

class LSTMLayer(object):

    def __init__(self,x,drop_masks):


        def forward_step(x_t, dropmask_t, prev_state,prev_content):
            input_gate =  T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_input),x_t) + T.dot(self.W_input,prev_state))
            forget_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_forget),x_t) + T.dot(self.W_input,prev_state))
            output_gate = T.nnet.hard_sigmoid(T.dot((dropmask_t * self.U_output),x_t) + T.dot(self.W_input,prev_state))

            stabilized_input = T.tanh(T.dot(((dropmask_t * self.U),x_t)) + T.dot(self.W,prev_state))

            memory_content = forget_gate * prev_content + input_gate * stabilized_input
            hidden_state = output_gate * T.tanh(memory_content)


            return [hidden_state,memory_content]



        [hidden_state,memory_content] = theano.scan(
            forward_step,
            sequences= [x,drop_masks],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])


        self.output = hidden_state




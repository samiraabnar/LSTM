import sys
sys.path.append('../../')
from LSTM.src.LSTMNetwork import *

if __name__ == '__main__':
    hidden_units = 100
    LSTMNetwork.train_1layer_glove_wordembedding(hidden_dim=hidden_units, modelfile="../data/LSTM/model_1/exp1_dc5h"+str(hidden_units)+".txt")

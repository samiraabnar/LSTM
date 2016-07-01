import numpy as np
import funcy
from funcy import project
import pickle


import sys
sys.path.append('../../')
from Util.util.math.MathUtil import *


class WordEmbeddingLayer(object):

    def __init__(self):
        self.word2vec = {}
        self.vec2word = {}


    def load_embeddings_from_glove_file(self,filename,filter):
        self.word2vec = {}
        self.vec2word = {}

        with open(filename,'r') as gfile:
            for line in gfile:
                parts = line.split()
                i = 0
                word = ''
                while not MathUtil.is_float(parts[i]):
                    word += " "+parts[i]
                    i += 1
                word = word.strip()
                if word in filter:
                    vector = [float(p) for p in parts[i:]]
                    vector = np.asarray(vector)
                    self.word2vec[word] = vector
                    self.vec2word[vector.tostring()] = word
        self.word2vec['UNK'] = np.zeros(self.word2vec["."].shape)
        self.vec2word[self.word2vec['UNK'].tostring()] = "UNK"



    def save_embedding(self,filename):
        with open(filename+"_word2vec.pkl","wb") as f:
            pickle.dump(self.word2vec,f)

        with open(filename+"_vec2word.pkl","wb") as f:
            pickle.dump(self.vec2word,f)


    def load_filtered_embedding(self,filename):
        with open(filename+"_word2vec.pkl","rb") as f:
            self.word2vec = pickle.load(f)

        with open(filename+"_vec2word.pkl","rb") as f:
            self.vec2word = pickle.load(f)

    def get_vector(self,word):
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            return self.word2vec['UNK']

    def get_word(self,vector):
        return self.vec2word[vector.tostring()]

    def filter_unseen_vocab(self,vocab):
        self.word2vec = project(self.word2vec, vocab)
        self.vec2word = project(self.vec2word, [self.word2vec[word].tostring() for word in vocab])

    @staticmethod
    def load_embedded_data(path, name, representation):
        embedded, labels = [], []

        with open(path+"embedded_"+name+"_"+representation+".pkl", "rb") as f:
            embedded = pickle.load(f)
        with open(path+"labels_"+name+".pkl", "rb") as f:
            labels = pickle.load(f)

        return np.asarray(embedded), np.asarray(labels)

    def embed_and_save(self,sentences,labels,path,name,representation):
        embedded = self.embed(sentences)

        with open(path+"embedded_"+name+"_"+representation+".pkl","wb") as f:
            pickle.dump(embedded,f)
        with open(path+"labels_"+name+".pkl", "wb") as f:
            pickle.dump(labels,f)

    def embed(self,sentences):
        embedded_sentences = []
        for sentence in sentences:
            embedded_sentences.append([self.get_vector(vocab) for vocab in sentence])

        return embedded_sentences











if __name__ == '__main__':
    wem = WordEmbeddingLayer()
    wem.load_embeddings_from_glove_file(filename="../data/glove.840B.300d.txt",filter=["Bye", "Ablah", "."])
    a = wem.get_vector("Hi")
    print(wem.get_word(a))


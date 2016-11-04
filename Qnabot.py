#-- encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

#-- import modules & set up logging
from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
import gbpython as gb
import gbNN
from url2text import url2text
from text2sentence import text2sentence, process_line
from word2vec_train import file_iterable
import cPickle

class Qnabot(object):

    def __init__(self, n_features, n_categories, size_hid):
        self.n_features = n_features
        self.n_categories = n_categories
        self.size_hid = size_hid

        self.file_text = None
        self.visited_urls = None

        self.file_sentences = None

        self.file_w2v = None

        self.file_text_Qs = None
        self.file_text_As = None

        self.file_NN = None

        self.NN = gbNN.gbNN(n_features, size_hid, n_categories)
        self.w2v = None


    def grab_text(self, url, file_text, countlimit=1, 
                  visited=None, BFS=False):
        if visited is None:
            visited = self.visited_urls
        url2text(url, filetext, countlimit, visited, BFS)
        self.file_text = file_text
        self.visited_urls = visited


    def gen_sentences(self, file_sentences, 
                      file_text=None, len_thres=5):
        if file_text is None:
            file_text = self.file_text
        self.file_sentences = file_sentences
        text2sentence(file_text, file_sentences, len_thres)


    def word2vec_train(self, file_sentences=None, 
                       min_count=5, workers=1):
        if file_sentences is None:
            file_sentences = self.file_sentences
        sentences = file_iterable(file_sentences)
        if self.w2v is None:
            self.w2v = Word2Vec(sentences, min_count=min_count, 
                                size=self.n_features, workers=workers)
        else:
            self.w2v.train(sentences)

    
    def word2vec_save(self, file_w2v):
        self.file_w2v = file_w2v
        self.w2v.save(file_w2v)

    def word2vec_load(self, file_w2v):
        if file_w2v is None:
            file_w2v = self.file_w2v
        self.w2v = Word2Vec.load(file_w2v)

    def sentence2vec(self, sentence):
        """Convert processed sentence to vector.
    
        w2v_model: a trained Word2Vec model
        sentence: a list of words forming a sentence
        Return: a list of float representing the vector
        """
        vec = np.zeros((self.n_features,), float)
        for word in sentence:
            try:
                vec += np.array(self.w2v[word])
            except KeyError:
                continue
        return vec/np.linalg.norm(vec)

    
    def NN_train(self, file_text_Qs, file_text_As, 
                 regul=0, maxiter=1000, verbose=False):
        self.file_text_Qs = file_text_Qs
        self.file_text_As = file_text_As
        #-- prepare X and y
        X, y = [], []
        #--- m = # of training examples
        m = 0
        if verbose:
            print 'Reading training examples'
        with open(file_text_Qs) as f:
            while True:
                line = gb.skipcomment(f, '#')
                if len(line) == 0: break
                line = line.split()
                #-- len(line)==1 means there is only 
                #--   one number (y) in the line
                if len(line) <= 1:
                    continue
                y.append(int(line[0]))
                sentence = ' '.join(line[1:])
                sentence = process_line(sentence)
                X.append(self.sentence2vec(sentence.split()))
        
        if verbose:
            print 'Preparing X and y'
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print 'Training'
        self.NN.train(X, y, regul=regul, verbose=verbose, 
                      maxiter=maxiter)

    def NN_save(self, file_NN):
        self.file_NN = file_NN
        cPickle.dump(self.NN, open(file_NN, 'w'))

    def NN_load(self, file_NN=None):
        if file_NN is None:
            file_NN = self.file_NN
        self.NN = cPickle.load(open(file_NN))
        
    def predict(self, Qs, n_res=1):
        #-- Prepare output list
        list_As = []
        with open(self.file_text_As) as f:
            while True:
                line = gb.skipcomment(f, '#')
                if len(line) == 0: break
                line = ' '.join(line.split()[1:])
                list_As.append(line)

        X = []
        for text in Qs:
            sentence = process_line(text)
            vec = self.sentence2vec(sentence.split())
            X.append(vec)
        if n_res == 1:
            y = self.NN.predict(np.array(X))
            ans = [list_As[idx] for idx in y]
        else:
            y = self.NN.predict_ranked(np.array(X), n_res)
            ans = [[None for j in xrange(y.shape[1])]
                         for i in xrange(y.shape[0])]
            for i in xrange(y.shape[0]):
                for j in xrange(y.shape[1]):
                    ans[i][j] = list_As[y[i][j]]
        return y, ans

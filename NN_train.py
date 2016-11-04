#-- encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import gbNN
import gbpython as gb
import text2sentence
from gensim.models import Word2Vec
import numpy as np
import cPickle
from sentence2vec import sentence2vec

#-- import modules & set up logging
from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

w2v_model = Word2Vec.load('word2vec_model_plusQs.sav')
#-- prepare X and y
X, y = [], []
#--- m = # of training examples
m = 0
print 'Reading training examples'
with open('raw_text_Qs.txt') as f:
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
        sentence = text2sentence.process_line(sentence)
        X.append(sentence2vec(w2v_model, sentence.split()))

print 'Preparing X and y'
X = np.array(X)
y = np.array(y)

nn_model = gbNN.gbNN(100, 100, 50)

print 'Training'
nn_model.train(X, y, regul=0.1, verbose=True, maxiter=1000)

# print 'Saving trained NN to disk'
# cPickle.dump(nn_model, open('NN.pickle', 'w'))



import gbNN
import gbpython as gb
from text2sentence import process_line
from gensim.models import Word2Vec
import numpy as np
import cPickle
from sentence2vec import sentence2vec

#-- Prepare models
w2v_model = Word2Vec.load('word2vec_model_plusQs.sav')
nn_model = cPickle.load(open('NN.pickle'))

#-- Prepare output list
list_As = []
with open('raw_text_As.txt') as f:
    while True:
        line = gb.skipcomment(f, '#')
        if len(line) == 0: break
        line = ' '.join(line.split()[1:])
        list_As.append(line)
        

# text = raw_input('Question: ')
Qs = ['how should i define a function', 
      'What are the boolean constants',
      'how to do power operation']
# Qs = []
# for line in open('sentences_Qs.txt'):
#     Qs.append(line)

X = []
for text in Qs:
    sentence = process_line(text)
    vec = sentence2vec(w2v_model, sentence.split())
    X.append(vec)

y = nn_model.predict_ranked(np.array(X), 5)
for i in xrange(y.shape[1]):
    print 'Question: %s'%(Qs[i],)
    print 'Best-matched answer: %s'%(list_As[y[0][i]],)
    print 'Other possible answers:'
    for j in xrange(1, y.shape[0]):
        print list_As[y[j][i]]
    print

# y = nn_model.predict(np.array(X))
# print y
# for i in xrange(len(y)):
#     print "Q: %s\nA: %s\n"%(Qs[i], list_As[y[i]])


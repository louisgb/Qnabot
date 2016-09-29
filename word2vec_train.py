#-- encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

#-- import modules & set up logging
from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class file_iterable(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename):
                yield line.split()

sentences = file_iterable('sentences.txt')
sentences_Qs = file_iterable('sentences_Qs.txt')
# model = Word2Vec(sentences, min_count=5, size=100, workers=8)
model = Word2Vec.load('word2vec_model.sav')
# model.build_vocab(sentences_Qs, keep_raw_vocab=True)
model.train(sentences_Qs)
model.save('word2vec_model_plusQs.sav')

# model.save('word2vec_model.sav')

# for line in open('sentences.txt'):
#     print line.split()
#     model.build_vocab(line.split())
#     model.train(line.split())



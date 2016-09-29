# encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

# import modules & set up logging
from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec.load('word2vec_model.sav')

print model['what']

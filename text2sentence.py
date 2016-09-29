# encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import re
from nltk import PorterStemmer

def process_line(line):
    """Process line for later feature extraction.

    line: a string representing a sentence.
    """
    words_exlude = set(['an','the','can','could','will','would','should','shall'])
    words_negative = set(['isn','aren','wasn','weren','ain','cannot','couldn','won','wouldn','shouldn','hasn','haven','hadn'])
    newline = line.lower()
    #-- remove URL
    newline = re.sub(u'(http|https)://[^\s]*', u' ', newline)
    #-- remove email address
    newline = re.sub(u'[^\s]+@[^\s]+', u' ', newline)
    #-- keep only letters
    newline = re.sub(u'[^a-zA-Z]', u' ', newline)
    lineout = ''
    for word in newline.split():
        if len(word)>1 and word not in words_exlude:
            if word in ['is','are','was','were','am']:
                word = 'be'
            elif word in ['has','had']:
                word = 'have'
            elif word in ['that','these','those']:
                word = 'this'
            elif word in words_negative:
                word = 'not'

            word = PorterStemmer().stem_word(word)
            # if word[-1]=='s': word = word[:-1]
            lineout += word+' '
    return lineout[:-1]
    

def text2sentences(filein, fileout, len_thres=5):
    """A wrapper that runs process_line on every line of a file.

    filein: input file name string
    fileout: output file name string
    len_thres: length threshold, lines shorter than which 
               are ignored.
    """
    fin = open(filein, 'r')
    fout = open(fileout, 'w')
    visited = set()

    for line in fin:
        newline = process_line(line)
        if len(newline.split()) > len_thres \
           and newline not in visited:
            visited.add(newline)
            print >>fout, newline

if __name__=='__main__':
    # filein = 'raw_text.txt'
    # fileout = 'sentences.txt'
    # text2sentences(filein, fileout)
    
    filein = 'raw_text_Qs.txt'
    fileout = 'sentences_Qs.txt'
    text2sentences(filein, fileout, 0)

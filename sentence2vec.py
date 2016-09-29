import numpy as np
from gensim.models import Word2Vec


def sentence2vec(w2v_model, sentence):
    """Convert processed sentence to vector.

    w2v_model: a trained Word2Vec model
    sentence: a list of words forming a sentence
    Return: a list of float representing the vector
    """
    vec = np.array(w2v_model[sentence[0]])
    for word in sentence[1:]:
        try:
            vec += np.array(w2v_model[word])
        except KeyError:
            continue
    return vec/np.linalg.norm(vec)


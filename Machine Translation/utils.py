import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cosine_similarity(u, v):
    """
    compute the cosine similarity score between two row vectors

    Params:
    ----------
    u: numpy array
        a n-dimensional row vector
    v: numpy array
        a n-dimensional row vector

    Returns: 
    ----------
    score: float
        the cosine similarity score of u and v
    """
    score = np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v))
    return score


def distance_cosine_score(u, v):
    score = 1 - np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v))
    return score


def get_word_to_indices_dict(en_fr_dict):
    """
    Returns the word to indices dictionary of english and french words

    Params:
    ----------
    en_fr_dict: dict
        dictionary mapping of english to french words
    
    Returns:
    ----------
    en_wi: dict
        word to indices dictionary of english words.
    fr_wi: dict
        word to indices dictionary of french words.
    """

    en_wi, fr_wi = {}, {}
    for idx, (en_word, fr_word) in enumerate(en_fr_dict.items()):
        en_wi[en_word] = idx
        fr_wi[fr_word] = idx
    return en_wi, fr_wi


def get_indices_to_word_dict(en_fr_dict, en_embeddings, fr_embeddings):
    """
    Returns the indices to word dictionary of english and french words

    Params:
    ----------
    en_fr_dict: dict
        Dictionary mapping of english to french words.
    en_embeddings: dict
        English words embedding dictionary.
    fr_embeddings: dict
        French words embedding dictionary.
    
    Returns:
    ----------
    en_iw: dict
        indices to word dictionary of english words.
    fr_iw: dict
        indices to word dictionary of french words.
    """

    en_vocab = set(en_embeddings.keys())
    fr_vocab = set(fr_embeddings.keys())

    en_iw, fr_iw = {}, {}
    idx = 0
    for en_word, fr_word in en_fr_dict.items():
        if en_word in en_vocab and fr_word in fr_vocab:
            en_iw[idx] = en_word
            fr_iw[idx] = fr_word
            idx += 1
    return en_iw, fr_iw


def get_en_to_fr_dict(filename, delimiter=' '):
    """
    Returns the python dictionary mapping from english to french words given the mapping in comma, space delimited file

    Params:
    ----------
    filename: str
        File containing the en-fr mapping in comma, space delimited format.
    delimiter: str
        Defines the delimiter format to use.
    
    Returns:
    ----------
    en_fr_dict: dict
        The python dictionary object containing the mapping of english to french words.
    """

    df = pd.read_csv(filename, delimiter=delimiter, header=None)
    
    en_fr_dict = {}
    for idx in range(len(df)):
        en_word = df.iloc[idx][0]
        fr_word = df.iloc[idx][1]
        en_fr_dict[en_word] = fr_word

    return en_fr_dict

def  get_word_embedding_matrices(en_fr_dict, en_embeddings, fr_embeddings):
    """
    Returns the matrices X and Y where each row in X is the word embedding for an english word,
    and the same row in Y is the word embedding for the French version of that English word.

    Params:
    ----------
    en_fr_dict: dict
        the python dictionary object containing the mapping of english to french words.
    en_embeddings: dict
        English words embedding dictionary.
    fr_embeddings: dict
        French words embedding dictionary.
    
    Returns:
    ----------
    X: numpy array
        English word embeddings in array form, where each row corresponds to the vector representation of the word
    Y: numpy array
        French word embeddings in array form, where each row corresponds to the vector representation of the word
    """

    # define the dimesion of english and french word embeddings
    dim = len(list(en_embeddings.values())[0])

    # build the vocabulary set of english and french words
    en_vocab = set(en_embeddings.keys())
    fr_vocab = set(fr_embeddings.keys())

    # initialize the X and Y matrices 
    X = np.array(np.zeros((1, dim)))
    Y = np.array(np.zeros((1, dim)))

    for en_word, fr_word in en_fr_dict.items():
        if en_word in en_vocab and fr_word in fr_vocab:
            X = np.vstack((X, en_embeddings[en_word]))
            Y = np.vstack((Y, fr_embeddings[fr_word]))
    
    return X[1:], Y[1:]


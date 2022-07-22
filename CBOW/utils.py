import re
import emoji
import numpy as np
from scipy import linalg
from nltk.tokenize import word_tokenize


def preprocess_data(filepath):
    """
    Load, tokenize, and process data.
    """
    with open(filepath, "r") as f:
        data = f.read()
    data = re.sub(r"[,!?;-]+", ".", data)           # replace punctuations with a period.                          
    data = word_tokenize(data)                      # tokenize the string data by word.
    data = [
        word.lower() for word in data if            # convert words to lowercase.
        word.isalpha() or                           # drop non-alphabetical tokens.
        emoji.is_emoji(word) or                     # keep emojis.
        word == '.'                                 # keep the period.
    ]
    return data


def get_word2ind_mapping_dict(data):
    """
    Returns the word to indices and indices to word mapping for a given data.
    """
    vocab = sorted(set(data))
    word2ind, ind2word = {}, {}
    for i in range(len(vocab)):
        word = vocab[i]
        word2ind[word] = i
        ind2word[i] = word
    return word2ind, ind2word


def softmax(z):
    """
    Computes the softmax for the given 2Darray.

    Params:
    ----------
    z: array of dimension (V x m).
        The batch output from the hidden CBOW layer.

    Returns:
    ----------
    Softmax of z.
    """
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)


def relu(h):
    """
    Computes the rectified linear unit (ReLU) activation function on given input batch 2Darray.

    Params:
    ----------
    h: array of dimension (N x m).
        The batch output from the hidden CBOW layer.

    Returns:
    ----------
    ReLU of h.
    """
    h[h < 0] = 0
    return h


def get_one_hot_vector(word, V, word2ind):
    """ 
    Returns the one hot vector for the input word.
    """
    vector = np.zeros((V, 1)) 
    vector[word2ind[word]] = 1
    return vector


def get_center_context_words(data, C, V, word2ind):
    """
    Generates One-hot Vectors for center and context words from the data.

    Params:
    ----------
    data: list
        List of words making up the training corpus.
    C: half context size.
    V: dimension of the vocabulary.
    word2ind: dict
        The word to indices dictionary.

    Returns:
    ----------
    X: array of dimension (N x 1)
        The average one hot vector represetation of the context words.
    Y: array of dimnesion (N x 1) 
        The one hot vector represetation of the center words.
    """

    center_word_idx = C
    while(center_word_idx < len(data) - C):
        # Get the center word for the current window.
        center_word = data[center_word_idx]
        # Get the One-Hot-Vector representation for the center word.
        center_word_vec = get_one_hot_vector(center_word, V, word2ind)

        # Get the context words for the current window.
        context_words = data[center_word_idx - C:center_word_idx] + data[center_word_idx + 1:center_word_idx + C + 1]

        # Get the One-Hot-Vector representation for the context words as
        # the average of the sum of the one-hot vectors of its constituent words.
        context_words_vec = np.zeros((V,1))
        for word in context_words:
            context_words_vec += get_one_hot_vector(word, V, word2ind) 
            context_words_vec = 1 / (2*C) * (context_words_vec)
        X = context_words_vec
        Y = center_word_vec
        # Generate samples.
        yield X, Y
        center_word_idx +=1
        # Repeat the training samples once all the samples are exhausted.
        # This results in an infinite streame of training samples.
        if(center_word_idx >= len(data) - C):
            center_word_idx = C


def get_batches(data, C, V, word2ind, batch_size=128):
    """
    Generates batches of data (X, Y) for the model to train on.
    """
    X_batch, Y_batch = [], []
    for X, Y in get_center_context_words(data, C, V, word2ind):
        if len(X_batch) < batch_size:
            X_batch.append(X)
            Y_batch.append(Y)
            continue
        X_batch = np.hstack(X_batch)
        Y_batch = np.hstack(Y_batch)
        yield X_batch, Y_batch
        X_batch, Y_batch = [], []


def compute_pca(data, n_components=2):
    """
    Input: 
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output: 
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    ### START CODE HERE ###
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]
    ### END CODE HERE ###
    return np.dot(evecs.T, data.T).T




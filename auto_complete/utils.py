import random
import nltk


def split_to_sentences(data):
    """
    Tokenizes a text corpus to sentences by linebreak "\\n" and 
    removes any white spaces before or after each sentences.

    Params:
    -----------
    data: str
        The text corpus.
    
    Returns:
    -----------
    sentences: list
        List of sentences.
    """
    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0 ]   # exclude whitespaces if any as sentences.

    return sentences


def tokenize_sentences(sentences):
    """
    Tokenize sentences to words.
    
    Params:
    -----------
    sentences: list
        List of sentences.
    
    Returns:
    -----------
    tokenized_sentences: list
        List of sentences tokenized to words.
    """

    tokenized_sentences = []
    for sent in sentences:
        tokenized_sentences.append(nltk.word_tokenize(sent.lower()))

    return tokenized_sentences


def train_eval_test_split(sentences, train=0.8, test=0.2, eval=0, random_seed=100):
    """
    Split data (sentences) into train, evaluation, and test set.
    
    Params:
    -----------
    sentences: list
        List of sentences.
    train: float
        Fraction of data to consider for training data.
    eval: float
        Fraction of data to consider for evaluation data.
    test: float
        Fraction of data to consider for test data.
    
    Returns:
    -----------
    train_data: list
        List of training data.
    eval_data: list
        List of evaluation data.
    test_data: list
        List of test data.
    """
    random.seed(random_seed)

    # randomly shuffle the data
    random.shuffle(sentences)

    N = len(sentences)

    train_data_len = int(N * train)
    test_data_len = int(N * test)
    # eval_data_len = int(N * eval)
    
    train_data = sentences[:train_data_len]
    test_data = sentences[train_data_len: train_data_len + test_data_len]
    eval_data = sentences[train_data_len + test_data_len: ]

    return train_data, test_data, eval_data


def create_vocab(data, freq=2):
    """
    Create a vocabulary of words that appear at least 'freq' number of times from the input data.

    Params:
    ----------
    data: list
        List of sentences containing list of word tokens.
    freq: int
        A word must occur at least these number of times in the input data to be considered.

    Returns:
    ----------
    vocab: list
        The vocabulary containing a list of words.
    """
    word_count_dict = {}
    for sent in data:
        for word in sent:
            word_count_dict[word] = word_count_dict.get(word, 0) + 1

    vocab = list(set(word for word, count in word_count_dict.items() if count >= freq))
    return vocab


def replace_oov_words_by_unk(tokenized_sentences, vocab, unknown_token='<unk>'):
    """
    Returns the input dataset after handling out of vocabulary(oov) words by 
    replacing such words with unknown character.

    Params:
    -----------
    tokenized_sentences: list
        List of sentences tokenized to words.
    vocab: list
        The list of words making up the vocabulary.
    unknown_token: str
        The token character which will be used to replace words in 
        the input dataset not present in the vocabulary.
    
    Returns:
    -----------
    tokenized_sentences_oov: list
        List of sentences after handling OOV words.
    """
    vocab = set(vocab)

    tokenized_sentences_oov = []
    for sent in tokenized_sentences:
        word_tokens = []
        for word in sent:
            if word not in vocab:
                word_tokens.append(unknown_token)
            else: word_tokens.append(word)
        tokenized_sentences_oov.append(word_tokens)
    return tokenized_sentences_oov
             




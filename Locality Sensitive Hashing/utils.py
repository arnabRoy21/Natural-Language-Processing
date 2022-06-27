import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """
    Processes a tweet by lowercasing, removing hastags, hyperlinks, punctuations 
    and returning a tokenized form (by words) of the tweet after removal of stopwords.

    Params:
    ----------
    tweet: str
        The tweet to process.

    cleaned_tweet: list
        Preprocessed and tokenized form of the input tweet.
    """

    hyperlink_pattern = r'http[s]?:\/\/\S+'
    processed_tweet = re.sub(hyperlink_pattern, '', tweet)    # remove hyperlinks

    processed_tweet = re.sub(r'^RT[\s]+', '', processed_tweet)          # remove old style retweet text "RT"

    hashtag_pattern = r'#'
    processed_tweet = re.sub(hashtag_pattern, '', processed_tweet)      # remove '#' tag

    stock_pattern = r'\$\w*'
    processed_tweet = re.sub(stock_pattern, '', processed_tweet)        # remove stock market tickers like $GE

    new_line_pattern = r'(\r\n|\r|\n)'
    processed_tweet = re.sub(new_line_pattern, '', processed_tweet)     # remove new lines and carriage return in the tweet
    
    tokenizer = TweetTokenizer(
        preserve_case=False,    # change tweet to lower case
        reduce_len=True,        # reduce sequence of chars greater than 3 or more to 3 
        strip_handles=True      # remove twitter handles like '@mike', '@john'
    )
    tokenized_tweet = tokenizer.tokenize(processed_tweet)

    en_stopwords = stopwords.words('english')
    punctuations = string.punctuation

    cleaned_tweet = []
    for word in tokenized_tweet:
        if word not in en_stopwords and word not in punctuations:
            cleaned_tweet.append(word)
    
    return cleaned_tweet


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
    """
    compute the distance cosine similarity score between two row vectors as d = 1 - cosine_similarity(u, v)

    Params:
    ----------
    u: numpy array
        a n-dimensional row vector
    v: numpy array
        a n-dimensional row vector

    Returns: 
    ----------
    score: float
        the distance cosine similarity score of u and v
    """
    score = 1 - np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v))
    return score

import re
import string


# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

# Define some Unknown word tokens
unk_tokens = [
    '--n--', '--unk_digit--', '--unk_punct--', 
    '--unk_upper--', '--unk_noun--', '--unk_verb--', 
    '--unk_adj--', '--unk_adv--', '--unk--'
    ]


def create_vocab(training_file_path):
    """
    Creates a vocabulary from the training corpus

    Params:
    ----------
    training_file_path: str
        Path to training file.
    
    Returns:
    ----------
    vocab: list
        List of words in the vocabulary
    """
    vocab = []

    # keep track of the count of the words
    count_dict = {}

    with open(training_file_path, 'r') as f:
        for line in f:
            if len(line.split()) == 2: 
                word = line.split()[0]
                count_dict[word] = count_dict.get(word, 0) + 1
    vocab = [word for word, count in count_dict.items() if count >= 2]
        
    # add unknown tokens in the vocab 
    vocab = vocab + unk_tokens
    return sorted(set(vocab))


def assign_unk(tok):
    """
    Assign unknown word tokens.

    Params:
    ----------
    tok: str
        The word token to check.
    
    Returns:
    ----------
    : str
        The Unknown tokens.
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"


def get_word_tag(line, vocab):
    """
    Returns the word and its associated POS tag from the given input string, the two delimited by \t or whitespaces.

    Params:
    ----------
    line: str
        A string containing a word and its associated tag, delimited by \t or whitespaces.

    vocab: list
        The vocabulary being used.

    Returns:
    ----------
    word: str
        A word.
    tag: str
        The tag associated with the word.
    """
    if len(line.split()) == 2:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
    else:
        # assign default values to invalid inputs
        word = '--n--'  # new line tag
        tag = '--s--'   # signifies start tag
    return word, tag


def process(sent):
    """
    Processes the given sentence into word tokens. All non-word characters are ignored/removed.

    Params:
    ----------
    sent: str
        The sentence to process.
    
    Returns:
    ----------
    word_tokens: list of words
        The list of word tokens.
    """
    pattern = r'\w+'
    word_tokens = re.findall(pattern, sent)
    return word_tokens


def build_word_index(vocab):
    """
    Builds the index of the words in the vocabulary.

    Params:
    ----------
    vocab: list of words
        The vocabulary being used.
    
    Returns:
    ----------
    word_index: dict
        A dictionary where the key is a word and the value is index of that word.
    """
    word_index = {}
    for i, word in enumerate(vocab):
        word_index[word] = i
    
    return word_index


def build_pos_tag_index(pos_states):
    """
    Builds the index of the POS tags.

    Params:
    ----------
    pos_states: list
        The list of POS tags.
    
    Returns:
    ----------
    pos_index: dict
        A dictionary where the key is a pos tag and the value is index of that tag.
    """
    pos_index = {}
    for i, pos in enumerate(pos_states):
        pos_index[pos] = i
    
    return pos_index

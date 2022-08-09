


def get_vocab_and_tags(words_path, tags_path):
    with open(words_path, 'r') as f:
        vocab = f.read().splitlines()
    with open(tags_path, 'r') as f:
        tags = f.read().splitlines()
    return vocab, tags


def build_word2idx_and_tag2idx_dict(vocab, tags):
    vocab_dict, tags_dict = {}, {}
    for word in vocab:
        if word not in vocab_dict: vocab_dict[word] = len(vocab_dict)
    vocab_dict['<PAD>'] = len(vocab_dict)
    
    for tag in tags:
        if tag not in tags_dict: tags_dict[tag] = len(tags_dict)
    
    return vocab_dict, tags_dict


def to_tensor(sentences_path, labels_path, vocab_dict, tag_dict):
    sentences_tensor, labels_tensor = [], []
    
    with open(sentences_path, 'r') as f:
        sentence_list = f.readlines()
    for sentence in sentence_list:
        sentence_tensor = [vocab_dict.get(word, vocab_dict.get('UNK')) 
                                          for word in sentence.strip().split()]
        sentences_tensor.append(sentence_tensor)
    
    with open(labels_path, 'r') as f:
        labels_list = f.readlines()
    for labels in labels_list:
        label_tensor = [tag_dict.get(tag, tag_dict.get('O')) 
                                          for tag in labels.strip().split()]
        labels_tensor.append(label_tensor)

    return sentences_tensor, labels_tensor
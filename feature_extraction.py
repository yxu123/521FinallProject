import numpy as np
from nltk import word_tokenize
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def parse(data_file):
    labels=[]
    texts=[]
    with open(data_file,'r') as data:
        for line in data:
            fields = line.split("\t")
            texts.append(fields[0])
            labels.append(fields[1])
    return labels,texts


def labels_to_key(labels):
    label_set = set(labels)
    label_key = {}
    for i, label in enumerate(label_set):
        label_key[label] = i
    return label_key


def labels_to_y(labels, label_key):

    y = np.zeros(len(labels), dtype=np.int)
    for i,l in enumerate(labels):
        y[i] = label_key[l]
    return y

def shuffle_dataset(data0, data1):
    """
    Shuffles two iterables containing associated data in unison, e.g. X and y; X and file id's
    :param data0: iterable, e.g. X
    :param data1: iterable, e.g. y
    :return: tuple (shuffled0, shuffled1)
    """
    # seed random for consistency in student homework
    np.random.seed(521)
    # define a new order for the indices of data0
    new_order = np.random.permutation(len(data0))

    # cast inputs to np array
    data0 = np.asarray(data0)
    data1 = np.asarray(data1)

    # reorder
    shuffled0 = data0[new_order]
    shuffled1 = data1[new_order]
    return (shuffled0, shuffled1)

def split_data(data0, data1, test_percent = 0.3, shuffle=True):
    """
    Splits dataset for supervised learning and evaluation
    :param data0: iterable, e.g. X, features
    :param data1: iterable, e.g. y, labels corresponding to the features in X
    :param test_percent: percent data to assign to test set
    :param shuffle: shuffle data order before splitting
    :return: two tuples, (data0_train, data1_train), (data0_test, data1_test)
    """
    if shuffle:
        data0, data1 = shuffle_dataset(data0, data1)
    data_size = len(data0)
    num_test = int(test_percent * data_size)

    train = (data0[:-num_test], data1[:-num_test])
    test = (data0[-num_test:], data1[-num_test:])
    return train, test


####w2v
def preprocess_text(text, stem=False):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a list of strings."""
    stops = set(stopwords.words('english'))
    toks = word_tokenize(text)
    if stem:
        stemmer = PorterStemmer()
        toks = [stemmer.stem(tok) for tok in toks]
    toks_nopunc = [tok for tok in toks if tok not in string.punctuation]
    toks_nostop = [tok for tok in toks_nopunc if tok not in stops]
    return toks_nostop

def w2v(word2vec_file,texts):
    vectors=np.zeros(len(texts),len())
    w2v_vectors = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    toks = preprocess_text(texts)
    veclist = [w2v_vectors[tok] for tok in toks if tok in w2v_vectors]


# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
    output_values = [row for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

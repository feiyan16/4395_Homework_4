# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import nltk
from nltk.util import ngrams
import pickle


def create_dict(n_grams):
    dictionary = {}
    for ngram in n_grams:
        if ngram not in dictionary:
            dictionary[ngram] = 1
        else:
            dictionary[ngram] += 1
    return dictionary


def read_file(name):
    file = open(name, 'r')
    text = ''
    for line in file.readlines():
        text += line[:len(line) - 1]
    unigrams = nltk.word_tokenize(text)
    bigrams = list(ngrams(unigrams, 2))
    uni_dict = create_dict(unigrams)
    bi_dict = create_dict(bigrams)
    file.close()
    return uni_dict, bi_dict


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filepaths = ['data/LangId.train.English', 'data/LangId.train.French', 'data/LangId.train.Italian']
    for filepath in filepaths:
        unigram_dict, bigram_dict = read_file(filepath)
        uni_filename = filepath + '_uni.p'
        bi_filename = filepath + '_bi.p'
        pickle.dump(unigram_dict, open(uni_filename, 'wb'))
        pickle.dump(bigram_dict, open(bi_filename, 'wb'))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

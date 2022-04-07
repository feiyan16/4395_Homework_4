import pickle
import nltk
from nltk.util import ngrams


def calc_vocab(uni_dicts):
    n = 0
    for uni_d in uni_dicts:
        n += len(uni_d)
    return n


def calc_prob(text, uni_counts, bi_counts, v):
    tokens = nltk.word_tokenize(text)
    bigrams = list(ngrams(tokens, 2))
    p_line = 1
    for bigram in bigrams:
        n = bi_counts[bigram] if bigram in bi_counts else 0
        d = uni_counts[bigram[0]] if bigram[0] in uni_counts else 0
        p_line *= ((n + 1) / (d + v))
    return p_line


def calc_accuracy(solutions, results):
    size = len(solutions)
    ct = 0
    for i in range(size):
        if solutions[i] != results[i]:
            print(i + 1)
            ct += 1
    print('Accuracy:', ((size - ct) / size) * 100)


def determine_language(lines, unigrams, bigrams, v):
    probabilities = {'English': 0, 'French': 0, 'Italian': 0}
    languages = ['English', 'French', 'Italian']
    results = open('data/LangId.result', 'a')
    for idx in range(len(lines)):
        max_prob = -1
        for i in range(len(languages)):
            probability = calc_prob(lines[idx], unigrams[i], bigrams[i], v)
            probabilities[languages[i]] = probability
            max_prob = max(max_prob, probability)
        ml_lang = ''
        for language in languages:
            if probabilities[language] == max_prob:
                ml_lang = language
                # results.write(str(idx + 1) + ' ' + language + '\n')
                # break
        results.write(str(idx + 1) + ' ' + ml_lang + '\n')
    results.close()


if __name__ == '__main__':
    filepaths_uni = ['data/LangId.train.English_uni.p', 'data/LangId.train.French_uni.p', 'data/LangId.train'
                                                                                          '.Italian_uni.p']
    filepaths_bi = ['data/LangId.train.English_bi.p', 'data/LangId.train.French_bi.p',
                    'data/LangId.train.Italian_bi.p', ]

    dict_uni = []
    dict_bi = []
    for path in filepaths_uni:
        dict_uni.append(pickle.load(open(path, 'rb')))
    for path in filepaths_bi:
        dict_bi.append(pickle.load(open(path, 'rb')))
    vocab_size = calc_vocab(dict_uni)

    test_file = open('data/LangId.test', 'r')
    determine_language(test_file.readlines(), dict_uni, dict_bi, vocab_size)
    test_file.close()

    sol_file = open('data/LangId.sol', 'r')
    result_file = open('data/LangId.result', 'r')
    calc_accuracy(sol_file.readlines(), result_file.readlines())
    result_file.close()
    sol_file.close()

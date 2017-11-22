from keras.models import load_model

import time

import numpy as np


def file_by_line(file, *args, **kwargs):
    with open(file, 'r', *args, **kwargs) as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


def log(tag, what=''):
    print('{} [{}] {}'.format(time.asctime(time.localtime(time.time())), tag, what))


def read_vocabulary(vocabulary_file):
    words = {}
    i = 0
    for line in file_by_line(vocabulary_file, errors='ignore'):
        line_parts = line.strip().split()

        if len(line_parts) != 3:
            continue

        (n, word, count) = line_parts
        words[word] = int(n)

        i = i + 1
    return words


def invert_vocabulary(vocabulary):
    inv = {}

    for (word, index) in vocabulary.items():
        inv[index] = word

    return inv


def get_nn_result(model, vocabulary, word_number):
    w1 = np.zeros((len(vocabulary),))
    w2 = np.zeros((len(vocabulary),))

    results = {}

    for w in range(1, len(vocabulary) + 1):
        w1[w - 1,] = word_number
        w2[w - 1,] = w

    out = model.predict([w1, w2])

    results = [(w, out[w - 1][0]) for w in range(1, len(vocabulary) + 1)]

    return sorted(results, key=lambda item: item[1], reverse=True)


log('start')
vocabulary = read_vocabulary('data_10000/vocabulary.txt')
inv_vocabulary = invert_vocabulary(vocabulary)
m = load_model('data_10000/model.0.h5')

word = vocabulary['год']  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
result = get_nn_result(m, vocabulary, word)

for i in range(0, 30):
    print(inv_vocabulary[result[i][0]], result[i])
log('end')

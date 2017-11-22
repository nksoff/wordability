from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape, dot as dot_layer
from keras.layers.embeddings import Embedding

import glob
import time
import random

import numpy as np


def file_by_line(file, *args, **kwargs):
    with open(file, 'r', *args, **kwargs) as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


def log(tag, what=''):
    print('{} [{}] {}'.format(time.asctime(time.localtime(time.time())), tag, what))


def create_model(vocabulary_size, vector_size=300):
    inputs = [Input((1,)), Input((1,))]
    embedding = Embedding(vocabulary_size, vector_size, input_length=1, name='embedding')
    target = Reshape((vector_size, 1))(embedding(inputs[0]))
    context = Reshape((vector_size, 1))(embedding(inputs[1]))
    dot = dot_layer([target, context], axes=1)
    dot = Reshape((1,))(dot)
    output = Dense(1, activation='sigmoid')(dot)

    model = Model(inputs=inputs, outputs=output)
    return model


def train(train_file, model, model_file, limit=200000):
    log('train start', train_file)
    w1 = np.zeros((1,))
    w2 = np.zeros((1,))
    label = np.zeros((1,))

    i = 0
    for line in file_by_line(train_file):
        if (random.randint(0, 1000) % 3 > 1):
            continue

        (w1_str, w2_str, label_str) = line.strip().split()

        w1[0,] = int(w1_str)
        w2[0,] = int(w2_str)
        label[0,] = int(label_str)

        loss = model.train_on_batch([w1, w2], label)

        if i % 1000 == 0:
            log('train line', '{} ({}, {}) => {} || loss: {}\n'.format(i, w1_str, w2_str, label_str, loss))
            model.save(model_file)

        if i > limit:
            log('train limit', '{} > {}'.format(i, limit))
            break
        i = i + 1

    model.save(model_file)
    log('train end', train_file)

    return model


log('start')
vocabulary_size = sum(1 for line in open('data_10000/vocabulary.txt')) + 2
vector_size = 500

train_files = sorted(glob.glob('data_10000/train.*.txt'))

i = 0
for train_file in train_files:
    model = create_model(vocabulary_size, vector_size)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    train(train_file, model, 'data_10000/model.{}.h5'.format(i), 200000)
    i = i + 1
log('end')

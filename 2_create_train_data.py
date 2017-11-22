from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence


def file_by_line(file, *args, **kwargs):
    with open(file, 'r', *args, **kwargs) as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


def log(tag, what=''):
    print('[{}] {}'.format(tag, what))


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_indexes(file):
    log('read_indexes start')
    indexes = []
    i = 0
    for line in file_by_line(file, errors='ignore'):
        line_indexes = map(int, line.strip().split())
        indexes.extend(line_indexes)

        if i % 10000 == 0:
            log('read_indexes line', i)

        i = i + 1
    log('read_indexes line', i)
    log('read_indexes end')

    return indexes


def read_vocabulary(vocabulary_file):
    log('read_vocabulary start')
    words = {}
    i = 0
    for line in file_by_line(vocabulary_file, errors='ignore'):
        line_parts = line.strip().split()

        if len(line_parts) != 3:
            continue

        (n, word, count) = line_parts
        words[word] = str(n)

        if i % 10000 == 0:
            log('read_vocabulary line', i)

        i = i + 1
    log('read_vocabulary line', i)
    log('read_vocabulary end')

    return words


def write_samples(output_file, data, vocabulary_size, sampling_table):
    log('write_samples start', output_file)
    couples, labels = skipgrams(data, vocabulary_size, window_size=3, sampling_table=sampling_table)
    with open(output_file, 'w') as f:
        for k in range(len(couples)):
            f.write('{} {} {}\n'.format(couples[k][0], couples[k][1], labels[k]))
    log('write_samples end', output_file)


log('start')
indexes = read_indexes('data_10000/indexes.txt')
vocabulary = read_vocabulary('data_10000/vocabulary.txt')
vocabulary_size = len(vocabulary) + 2

sampling_table = sequence.make_sampling_table(vocabulary_size)
i = 0
for data in chunks(indexes, 100000000):
    write_samples('data_10000/train.{}.txt'.format(i), data, vocabulary_size, sampling_table)
    i = i + 1
log('end')

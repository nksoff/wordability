def file_by_line(file, *args, **kwargs):
    with open(file, 'r', *args, **kwargs) as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


def log(tag, what=''):
    print('[{}] {}'.format(tag, what))


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


def write_words_indexes(dataset_file, vocabulary, output_file):
    log('write_words_indexes start')
    i = 0
    with open(output_file, 'w') as output:
        for line in file_by_line(dataset_file, errors='ignore'):
            line_words = line.strip().split()
            line_words_nums = list(map(lambda word: vocabulary.get(word, '0'), line_words))

            output.write('{}\n'.format(' '.join(line_words_nums)))

            if i % 10000 == 0:
                log('write_words_indexes line', i)

            i = i + 1
    log('write_words_indexes line', i)
    log('write_words_indexes end')


log('start')
vocabulary = read_vocabulary('data_10000/vocabulary.txt')
write_words_indexes('data_10000/data.txt', vocabulary, 'data_10000/indexes.txt')
log('end')

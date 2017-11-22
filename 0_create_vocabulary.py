from collections import Counter

stop_words = ['риа', 'тасс', 'сразу', 'между', 'лишь', 'при', 'для', 'перед', 'после', 'или', 'под', 'над', 'также',
              'если', 'из-за', 'через', 'потому', 'тоже', 'среди', 'всегда', 'какой-то', 'хотя', 'впрочем', 'якобы',
              'чтобы', 'как', 'так', 'когда', 'только', 'еще', 'ещë', 'уже', 'я', 'меня', 'мне', 'мной', 'мною', 'мы',
              'нас', 'нам', 'нами', 'ты', 'тебя', 'тебе', 'тобой', 'тобою', 'вы', 'вас', 'вам', 'вами', 'он', 'оно',
              'его', 'него', 'ему', 'нему', 'им', 'ним', 'нем', 'нём', 'она', 'ее', 'еë', 'нее', 'неё', 'ей', 'ней',
              'ею', 'нею', 'они', 'их', 'них', 'ими', 'ними', 'мой', 'мое', 'моë', 'моего', 'моему', 'моем', 'моëм',
              'моя', 'мою', 'моей', 'моею', 'мои', 'моих', 'моим', 'моими', 'наш', 'наше', 'нашего', 'нашему', 'нашим',
              'нашем', 'наша', 'нашу', 'нашей', 'нашею', 'наши', 'наших', 'нашими', 'кто', 'кого', 'кому', 'кем', 'ком',
              'что', 'чего', 'чему', 'чем', 'этот', 'это', 'этого', 'этому', 'этим', 'этом', 'эта', 'эту', 'этой',
              'этою', 'эти', 'этих', 'этими', 'тот', 'то', 'того', 'тому', 'том', 'та', 'ту', 'той', 'тою', 'те', 'тех',
              'тем', 'теми', 'такой', 'такое', 'такого', 'такому', 'таком', 'такая', 'такую', 'такою', 'такие', 'таких',
              'таким', 'такими', 'свой', 'свое', 'своё', 'своего', 'своему', 'своим', 'своем', 'своём', 'своя', 'свою',
              'своей', 'своею', 'свои', 'своих', 'своими', 'который', 'которое', 'которого', 'которому', 'которым',
              'котором', 'которая', 'которую', 'которой', 'которою', 'которые', 'которых', 'которыми', 'сам', 'себя',
              'себе', 'собой', 'собою', 'один', 'одно', 'одного', 'одному', 'одним', 'одном', 'одна', 'одну', 'одной',
              'одною', 'одни', 'одних', 'одними', 'весь', 'всë', 'всего', 'всему', 'всем', 'вся', 'всю', 'всей', 'всею',
              'все', 'всех', 'всеми', 'кто-то', 'что-то', 'быть', 'был', 'была', 'было', 'были', 'буду', 'будем',
              'будешь', 'будет', 'будут', 'будь', 'будьте', 'есть', 'мочь', 'могу', 'можем', 'можешь', 'можете',
              'может', 'могут', 'мог', 'могла', 'могло', 'могли', 'даже', ]


def is_valid_word(word):
    return len(word) > 2 and word not in stop_words


def file_by_line(file, *args, **kwargs):
    with open(file, 'r', *args, **kwargs) as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()


def log(tag, what=''):
    print('[{}] {}'.format(tag, what))


def create_counter(dataset_file):
    log('create_counter start')
    counter = Counter()
    i = 0
    for line in file_by_line(dataset_file, errors='ignore'):
        line_words = filter(is_valid_word, line.strip().split())
        counter.update(line_words)

        if i % 10000 == 0:
            log('create_counter line', i)
        i = i + 1
    log('create_counter line', i)
    log('create_counter end')

    return counter


def write_vocabulary(counter, output_file, number_of_words=10000):
    log('write_vocabulary start')
    i = 0
    with open(output_file, 'w') as f:
        for (word, count) in counter.most_common(number_of_words):
            f.write('{} {} {}\n'.format(i + 1, word, count))

            if i % 10000 == 0:
                log('write_vocabulary line', i)

            i = i + 1
    log('write_vocabulary line', i)
    log('write_vocabulary end')


log('start')
words_counter = create_counter('data_10000/data.txt')
write_vocabulary(words_counter, 'data_10000/vocabulary.txt', 10000)
log('end')
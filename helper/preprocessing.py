import codecs
import os
import numpy as np


def process_sentence(sentence : str) -> str:
    punct = [';', r"/", '[', ']', '"', '{', '}',
                    '(', ')', '=', '+', '\\', '_', '-',
                    '>', '<', '@', '`', ',', '?', '!',':','।']

    inText = sentence.replace('\n', ' ')
    inText = inText.replace('\t', ' ')
    inText = inText.strip()

    outText = inText

    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')

    outText=outText.split()
    outText = ' '.join(outText)

    return outText


def process_answer(text : str):
    manualMap = { 'শূন্য': '০', 'এক': '১', 'দুই': '২', 'তিন':
                '৩', 'চার': '৪', 'পাঁচ': '৫', 'ছয়': '৬', 'সাত': '৭',
                'আট': '৮', 'নয়': '৯', 'দশ': '১০'}

    new_answer = process_sentence(text)
    outText = []
    for word in new_answer.split():
        word = manualMap.setdefault(word, word)
        outText.append(word)

    return ' '.join(outText)


def load_glove_model(glove_file_path):
    if not os.path.isfile(glove_file_path):
        raise("Glove vector file not found")

    f = codecs.open(glove_file_path, 'r', encoding='utf-8')
    print('f:  ',f)
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    return model


def load_bangla_word2vec(bangla_vec_path):
    if not os.path.isfile(bangla_vec_path):
        raise("Bengali vector file not found")

    f = open(bangla_vec_path,'r')
    w2v_100d={}
    for i in f:
        lst=i.split()
        word=lst[0]
        word_vec=np.array(lst[1:101], dtype='float32')
        w2v_100d[word]=word_vec

    return w2v_100d
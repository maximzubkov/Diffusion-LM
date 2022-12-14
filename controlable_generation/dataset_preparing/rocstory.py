import csv
import json
from collections import Counter, defaultdict

import numpy as np
from spacy.lang.en import English

from improved_diffusion.rounding import load_tokenizer


def get_corpus_rocstory(data_args):
    '''

    :param data_args:  --> this is actually the model_args in the main function.
    :return:
    '''

    # print(data_args.task, 'DEBUG', '*---'*100)
    # print(model_args.task, 'DEBUG', '*---' * 100)
    if data_args.experiment.startswith('roc') and data_args.task == 'infill':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/ROCstory_full.csv', 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for idx, row in enumerate(roc_reader):
                if idx == 0:
                    continue
                sentences = row[2:]
                for ii in [1, 2, 3]:
                    sent = " ".join([sentences[ii-1], sentences[ii+1], sentences[ii]])
                    example = [x.text for x in tokenizer(sent)]
                    sentence_lst.append(example)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('roc') and data_args.task == 'classifier':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/ROCstory_full.csv', 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for idx, row in enumerate(roc_reader):
                if idx == 0:
                    continue
                sentences = row[2:]
                sentences = [[x.text for x in tokenizer(sent)] for sent in sentences]
                for ii in [1, 2, 3]:
                    # sent = " ".join([sentences[ii-1], sentences[ii+1], sentences[ii]])
                    example = [sentences[ii-1], sentences[ii+1], sentences[ii], 1]
                    sentence_lst.append(example)
        np.random.shuffle(sentence_lst)

        # construct negative examples/
        wrong_lst = []
        for idx, sent in enumerate(sentence_lst[:-1]):
            wrong_mid = sentence_lst[idx+1][2]
            wrong_tup = (sent[0], sent[1], wrong_mid, 0)
            wrong_lst.append(wrong_tup)

        sentence_lst = sentence_lst + wrong_lst

        print(sentence_lst[:2], sentence_lst[-2:])
        return sentence_lst, {}
    elif data_args.experiment.startswith('roc') and data_args.task != 'data_teacher':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/roc_train.json', 'r') as roc_reader:
            for row in roc_reader:
                sentences = json.loads(row)[0].strip()
                word_lst = [x.text for x in tokenizer(sentences)]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('roc') and data_args.task == 'data_teacher':
        print('loading dataset from ROCStory')
        sentence_lst = []
        with open(data_args.roc_train, 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for row in roc_reader:
                sentences = " ".join(row[2:])
                sentence_lst.append(sentences)
        sentence_lst = sentence_lst[1:]
        print(sentence_lst[:2])
        return sentence_lst, None
    elif data_args.experiment.startswith('simple-wiki'):
        print('loading dataset from simple wikipedia')
        sentence_lst = []
        with open(data_args.wiki_train, 'r') as ff:
            for row in ff:
                word_lst = row.split()
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'data_teacher':
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'finetuneUNK':
        '''
            Used to evaluate fluency: first load e2e-vocab, and then UNK the oov words in the training data. 
        '''
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        # load vocab.
        tokenizer2 = load_tokenizer('e2e-tgt', 'random',
                                   '/u/scr/nlp/xlisali/predictability/diffusion_models_v6/diff_e2e-tgt_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart')
        vocab = {v: k for k, v in tokenizer2.items()}
        print(len(tokenizer2), len(vocab), 'loaded vocabs')

        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                tokenized = [x.text for x in tokenizer(word_lst)]
                word_lst1 = [x if x in vocab else 'UNK' for x in tokenized]
                word_lst1 = " ".join(word_lst1)
                word_lst2 = [vocab.get(x.text, vocab['UNK']) for x in tokenizer(word_lst)]
                word_lst2 = " ".join([tokenizer2[x] for x in word_lst2])
                # print(word_lst1, word_lst2)
                assert word_lst1 == word_lst2

                # print(word_lst1)
                sentence_lst.append(word_lst1)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'right2left':
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                word_lst = list(reversed([x.text for x in tokenizer(word_lst)]))
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt'):
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                word_lst = [x.text for x in tokenizer(word_lst)]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('e2e-back'):
        ordered_ = ['name', 'Type', 'area', 'customer rating', 'near',
                    'family friendly', 'food', 'price']
        full_dict = defaultdict(lambda:Counter())
        def ordered_fill(src_lst, mode='full', full_dict=None):
            pair_lst = {x.split(':')[0].lstrip().strip():x.split(':')[1].lstrip().strip() for x in src_lst.split('|')}
            # print(pair_lst, 'hello')
            if mode == 'full':
                for x in ordered_:
                    v = pair_lst.get(x, 'none')
                    result_lst.append(f"{x} : {v}")
                return "|".join(result_lst)
            else:
                # print(pair_lst)
                v = pair_lst.get(mode, 'none')
                full_dict[mode][v] += 1
                # print(v)
                return f"{mode} : {v}"

        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        vocab_lst = []
        with open(path, 'r') as ff:
            for row in ff:
                src_lst, word_lst = row.split('||')
                # src_lst = ordered_fill(src_lst, 'food')
                # src_lst = ordered_fill(src_lst, 'price')

                word_lst = [x.text for x in tokenizer(word_lst)]
                for mode in ordered_:
                    src_lst3 = ordered_fill(src_lst, mode, full_dict)
                    src_lst2 = [x.text for x in tokenizer(src_lst3)]
                    sentence_lst.append((word_lst, src_lst2))
                vocab_lst.append(word_lst)

                # src_lst = ordered_fill(src_lst, 'area')
                # word_lst = [x.text for x in tokenizer(word_lst)]
                # src_lst = [x.text for x in tokenizer(src_lst)]
                # sentence_lst.append((word_lst, src_lst))
        print(sentence_lst[:2])
        print(full_dict)

        counter = Counter()
        for input_ids in vocab_lst:
            counter.update(input_ids)
            # counter.update(src_ids)

    # get tokenizer.
    if not data_args.experiment.startswith('e2e-back'):
        counter = Counter()
        for input_ids in sentence_lst:
            counter.update(input_ids)

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    for k, v in counter.items():
        if v > 10:
            vocab_dict[k] = len(vocab_dict)
    print(len(counter), len(vocab_dict))

    return sentence_lst, vocab_dict
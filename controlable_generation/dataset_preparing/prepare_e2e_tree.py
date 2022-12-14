import benepar

from controlable_generation.utils import _collate_batch_helper
from controlable_generation.utils import chart_from_tree, remove_leaves, pad_charts


def prepare_e2e_tree(raw_datasets, tokenizer, parser, tree_vocab, model_args):
    assert model_args.task != 'data_teacher', 'should not be data_teacher.'

    def tokenize_function(examples):
        vocab_dict = raw_datasets.vocab
        sent_lst = []
        for sent in examples['text']:
            input_sentence1 = benepar.InputSentence(
                words=sent[:63],
            )
            sent_lst.append(input_sentence1)
        parse_lst = list(parser.parse_sents(sent_lst))
        assert len(parse_lst) == len(examples['text'])

        if model_args.experiment == 'e2e-tgt-tree':
            chart_lst = []
            for x in parse_lst:
                chart = chart_from_tree(tree_vocab, x)
                chart_lst.append(chart)
            input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in
                         examples['text']]
            result_dict = {'input_ids': input_ids, 'chart_lst': chart_lst}
        elif model_args.experiment == 'e2e-tgt-gen-tree':
            parse_lst = [remove_leaves(tree) for tree in parse_lst]

            if model_args.task == 'finetune':
                input_strings = [
                    tree._pformat_flat("", "()", False) + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token
                    for
                    (tree, seq) in zip(parse_lst, examples['text'])]
                return tokenizer(input_strings, max_length=256, padding='max_length', truncation=True)
            elif model_args.task == 'from_scratch':
                raise NotImplementedError
                input_ids = [tree + [0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (tree, seq)
                             in
                             zip(parse_lst, examples['text'])]
        elif model_args.experiment == 'e2e-tgt-gen-spans':
            if model_args.task == 'finetune':
                input_strings = []
                for parse, seq in zip(parse_lst, examples['text']):
                    chart, spans = chart_from_tree(tree_vocab, parse, verbose=True)
                    for (a, b, c) in spans:
                        input_strings.append(
                            f"{a}, {b}, {c}" + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token)
                # print(len(input_strings), len(examples['text']))
                return tokenizer(input_strings, max_length=70, padding='max_length', truncation=True)
            elif model_args.task == 'from_scratch':
                input_lst = []
                for parse, seq in zip(parse_lst, examples['text']):
                    chart, spans = chart_from_tree(tree_vocab, parse, verbose=True)
                    for (a, b, c) in spans:
                        input_ids = [vocab_dict.get(x, vocab_dict['UNK']) for x in f"{a} {b} {c}".split()] + [0] + [
                            vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1]
                        input_lst.append(input_ids)
                print(len(input_lst), len(parse_lst))
                print(input_lst[0])
                result_dict = {'input_ids': input_lst}
        return result_dict

    def pad_function(group_lst):
        vocab_dict = raw_datasets.vocab
        if model_args.experiment == 'e2e-tgt-tree':
            max_length = 64
            group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
            group_lst['parse_chart'] = pad_charts(group_lst['chart_lst'], padding_value=-100)
        elif model_args.experiment == 'e2e-tgt-gen-tree' or model_args.experiment == 'e2e-tgt-gen-spans':
            if model_args.task == 'finetune':
                group_lst['labels'] = group_lst['input_ids']
            elif model_args.task == 'from_scratch':
                max_length = 64
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'],
                                                               max_length)
                group_lst['labels'] = group_lst['input_ids']

        return group_lst

    return tokenize_function, pad_function

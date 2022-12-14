from controlable_generation.utils import _collate_batch_helper


def prepare_e2e_back(raw_datasets, tokenizer, model_args):
    def tokenize_function(examples):
        vocab_dict = raw_datasets.vocab
        if model_args.experiment == 'e2e-back':
            input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (seq, _) in
                         examples['text']]
            src_ids = [[vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (_, seq) in examples['text']]
            result_dict = {'word_ids': input_ids, 'src_ids': src_ids}
            return result_dict
        elif model_args.experiment == 'e2e-back-gen':
            input_strings = [
                " ".join(attributes) + tokenizer.bos_token + " ".join(words) + tokenizer.eos_token
                for (words, attributes) in examples['text']]
            return tokenizer(input_strings, max_length=100, padding='max_length', truncation=True)

    def pad_function(group_lst):
        if model_args.experiment == 'e2e-back':
            vocab_dict = raw_datasets.vocab
            max_length = 64
            seqlen = 64
            group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
            max_src_length = max([len(xx) for xx in group_lst['src_ids']])
            max_src_length = min(seqlen, max_src_length)
            group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
                                                                                vocab_dict['PAD'],
                                                                                max_src_length,
                                                                                return_mask=True)

            group_lst['input_ids'] = [x + y for (x, y) in zip(group_lst['word_ids'], group_lst['src_ids'])]
            group_lst['labels'] = [[-100] * len(x) + y for (x, y) in zip(group_lst['word_ids'], group_lst['src_ids'])]
        elif model_args.experiment == 'e2e-back-gen':
            group_lst['labels'] = group_lst['input_ids']
        return group_lst
    return tokenize_function, pad_function
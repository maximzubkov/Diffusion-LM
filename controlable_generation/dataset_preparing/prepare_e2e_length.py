def prepare_e2e_length(raw_datasets, tokenizer, pos_vocab, model_args):
    assert model_args.task != 'data_teacher', 'should not be data_teacher.'

    def tokenize_function(examples):
        vocab_dict = raw_datasets.vocab
        if model_args.task == 'finetune':
            input_strings = [f'{len(seq)}' + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token
                             for seq in examples['text']]
            return tokenizer(input_strings, max_length=128, padding='max_length', truncation=True)
        elif model_args.task == 'from_scratch':
            raise NotImplementedError
            input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in
                         examples['text']]
            pos_tags = [[0] + [pos_vocab[x] for x in seq] + [1] for seq in pos_lst]
            result_dict = {'input_ids': input_ids, 'pos_tags': pos_tags}

        return result_dict

    def pad_function(group_lst):
        group_lst['labels'] = group_lst['input_ids']
        return group_lst

    return tokenize_function, pad_function

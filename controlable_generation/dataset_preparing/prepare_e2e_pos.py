import logging

import spacy_stanza
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
)

from controlable_generation.utils import _collate_batch_helper

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def prepare_e2e_pos(raw_datasets, tokenizer, pos_vocab, model_args):
    assert model_args.task != 'data_teacher', 'should not be data_teacher.'
    nlp = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})

    def tokenize_function(examples):
        vocab_dict = raw_datasets.vocab
        sent_lst = [" ".join(seq) for seq in examples['text']]
        sent_full = " ".join(sent_lst)
        doc = nlp(sent_full)
        doc_token_pos = [(token.text, token.pos_,) for token in doc]
        len_lst = [len(seq) for seq in examples['text']]

        assert sum(len_lst) == len(doc_token_pos)
        pos_lst = []
        init_idx = 0
        for len_temp in len_lst:
            pos_lst.append([x[1] for x in doc_token_pos[init_idx:init_idx+len_temp]])
            init_idx = init_idx+len_temp

        if model_args.experiment == 'e2e-tgt-pos':
            input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
            pos_tags = [[0] + [pos_vocab[x] for x in seq] + [1] for seq in pos_lst]
            result_dict = {'input_ids': input_ids, 'pos_tags':pos_tags}
        elif model_args.experiment == 'e2e-tgt-gen-pos':
            if model_args.task == 'finetune':
                input_strings = [" ".join(pos_) + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token
                                 for (pos_, seq) in zip(pos_lst, examples['text'])]
                return tokenizer(input_strings, max_length=128, padding='max_length', truncation=True)
            elif model_args.task == 'from_scratch':
                input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                pos_tags = [[0] + [pos_vocab[x] for x in seq] + [1] for seq in pos_lst]
                result_dict = {'input_ids': input_ids, 'pos_tags': pos_tags}

        return result_dict

    def pad_function(group_lst):
        if model_args.experiment == 'e2e-tgt-pos':
            vocab_dict = raw_datasets.vocab
            max_length = 64
            group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
            max_src_length = 64 #min(seqlen, max_src_length)
            group_lst['pos_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['pos_tags'],
                                                                                vocab_dict['PAD'], max_src_length, return_mask=True)
            group_lst['labels'] = [[-100] * len(x) + y for (x, y) in zip(group_lst['input_ids'], group_lst['pos_ids'])]
        elif model_args.experiment == 'e2e-tgt-gen-pos':
            group_lst['labels'] = group_lst['input_ids']

        return group_lst

    return tokenize_function, pad_function
import logging
import os
from dataclasses import dataclass, field
from os.path import join
from typing import Optional

import benepar
from datasets import Dataset
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from controlable_generation.dataset_preparing.prepare_e2e_back import prepare_e2e_back
from controlable_generation.dataset_preparing.prepare_e2e_length import prepare_e2e_length
from controlable_generation.dataset_preparing.prepare_e2e_pos import prepare_e2e_pos
from controlable_generation.dataset_preparing.prepare_e2e_tree import prepare_e2e_tree
from controlable_generation.dataset_preparing.rocstory import get_corpus_rocstory
from controlable_generation.models.ar import AR_for_cont
from controlable_generation.models.classifiers.consistency import Classifier_Consistency
from controlable_generation.models.classifiers.gpt2 import Classifier_GPT2
from controlable_generation.models.classifiers.pos import Classifier_POS
from controlable_generation.models.classifiers.tree import Classifier_Tree
from controlable_generation.utils import _collate_batch_helper, group_texts

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    experiment: Optional[str] = field(
        default='compress',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    learned_emb: Optional[str] = field(
        default='no',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    padding_mode: Optional[str] = field(
        default='block',
        metadata={"help": "blcok or pad"},
    )
    roc_train: Optional[str] = field(
        default=join("datasets", "ROCstory"),
        metadata={"help": "roc story path"},
    )
    wiki_train: Optional[str] = field(
        default='/u/scr/xlisali/diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
        metadata={"help": "simple wiki path"},
    )
    e2e_train: Optional[str] = field(
        default=join("datasets", "e2e_data"),
        metadata={"help": "simple wiki path"},
    )

    reduced_emb: Optional[int] = field(
        default=8,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    rounding_mode: Optional[str] = field(
        default='gpt2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    sigma: Optional[float] = field(
        default=1.0,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    n_embd: Optional[int] = field(
        default=16,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    init_emb: Optional[str] = field(
        default="",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    task: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    synth_config:  Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml',
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    train_dataset, vocab = get_corpus_rocstory(data_args)

    if model_args.task == 'classifier':
        train_dataset = list(zip(*train_dataset))
        train_datasets = Dataset.from_dict({
            'left_text': train_dataset[0],
            'right_text': train_dataset[1],
            'mid_text': train_dataset[2],
            'label': train_dataset[3]
        })
    else:
        train_datasets = Dataset.from_dict({'text': train_dataset})
    raw_datasets = train_datasets.train_test_split(0.01)

    if model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-gen-pos']:
        exp_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
        pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                   'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                   'PUNCT', 'SYM', 'X']
        for x in pos_lst:
            exp_vocab[x] = len(exp_vocab)
    elif model_args.experiment in ['e2e-tgt-tree', 'e2e-tgt-gen-tree', 'e2e-tgt-gen-spans']:
        parser = benepar.Parser("benepar_en3")
        exp_vocab = parser._parser.config["label_vocab"]

    raw_datasets.vocab = vocab
    raw_datasets['validation'] = raw_datasets['test']

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.task in ['data_teacher', 'finetune']:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        if model_args.experiment in ['e2e-tgt-gen-tree', 'e2e-tgt-gen-spans', 'e2e-tgt-gen-pos']:
            tokenizer.add_tokens(list(exp_vocab.keys()))
        elif model_args.experiment == 'e2e-tgt-gen-length':
            tokenizer.add_tokens([str(xx) for xx in range(64)])
        elif model_args.experiment == 'e2e-tgt' and model_args.task == 'finetune':
            tokenizer.add_tokens(['UNK'])
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    else:
        tokenizer = raw_datasets.vocab
        if model_args.experiment == 'e2e-tgt-gen-spans':
            for x in exp_vocab.keys():
                tokenizer[x] = len(tokenizer)
            for x in range(64):
                if str(x) not in tokenizer:
                    tokenizer[str(x)] = len(tokenizer)

    if model_args.experiment in [
        'e2e-tgt-gen-tree', 'e2e-tgt-gen-pos', 'e2e-back-gen', 'e2e-tgt-gen-length', 'e2e-tgt-gen-spans'
    ]:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
    elif model_args.experiment in ['e2e-back', 'e2e-tgt-pos', 'e2e-tgt-tree']:
        import torch
        import json
        from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        config_path = join(model_args.init_emb, "training_args.json")
        with open(config_path, 'rb', ) as f:
            training_args2 = json.load(f)
        training_args2['sigma_small'] = True
        training_args2['diffusion_steps'] = 200
        temp_dict = model_and_diffusion_defaults()
        temp_dict.update(training_args2)
        _, diffusion = create_model_and_diffusion(
            **temp_dict
        )

        config.vocab_size = len(tokenizer)
        config.input_emb_dim = model_args.n_embd
        config.train_diff_steps = 200

        if model_args.experiment == 'e2e-back':
            model = Classifier_GPT2(config=config, diffusion=diffusion)
        elif model_args.experiment == 'e2e-tgt-pos':
            config.pos_vocab_size = len(exp_vocab)
            model = Classifier_POS(config=config, diffusion=diffusion)
        elif model_args.experiment == 'e2e-tgt-tree':
            config.tree_vocab_size = len(exp_vocab)
            model = Classifier_Tree(config=config, diffusion=diffusion)

        filename = model_args.init_emb  # '/u/scr/nlp/xlisali/predictability/diffusion_models_v3/diff_e2e-tgt_block_rand16_transformer_lr0.0001_2000_cosine_Lsimple_h128_s2_sd101'
        path_save = '{}/random_emb.torch'.format(filename)
        path_learned = '{}/ema_0.9999_200000.pt'.format(filename)
        if model_args.learned_emb == "yes":
            learned_embeddings = torch.load(path_learned)['word_embedding.weight']
            if model_args.experiment in ['e2e-tgt-pos', 'e2e-back', 'e2e-tgt-tree']:
                model.transformer.embeddings.word_embeddings.weight.data = learned_embeddings.clone()
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
        else:
            if model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-tree']:
                model.transformer.embeddings.word_embeddings.load_state_dict(torch.load(path_save))
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
            elif model_args.experiment in ['e2e-back']:
                model.transformer.wte.load_state_dict(torch.load(path_save))
                model.transformer.wte.weight.requires_grad = False
    elif model_args.experiment in ['pos', 'synth', 'roc', 'simple-wiki', 'e2e-tgt']:
        if model_args.task == 'ar_for_cont':
            import torch
            config.sigma = model_args.sigma
            config.n_embd = model_args.n_embd
            config.n_head = model_args.n_embd
            config.vocab_size = len(tokenizer)
            model = AR_for_cont(config)
            filename = model_args.init_emb #'/u/scr/nlp/xlisali/predictability/diffusion_models_v3/diff_e2e-tgt_block_rand16_transformer_lr0.0001_2000_cosine_Lsimple_h128_s2_sd101'
            path_save = '{}/random_emb.torch'.format(filename)
            model.transformer.wte.load_state_dict(torch.load(path_save))
            model.transformer.wte.weight.requires_grad = False
            print(model.lm_head.weight)
            print(model.transformer.wte.weight)
        if model_args.task == 'data_teacher' or model_args.task == 'finetune':
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
            model.resize_token_embeddings(len(tokenizer))
        elif model_args.task == 'classifier':
            import torch
            import json
            from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

            config.vocab_size = len(tokenizer)
            config.type_vocab_size = 3

            config_path = os.path.join(model_args.init_emb, "training_args.json")
            with open(config_path, 'rb', ) as f:
                training_args2 = json.load(f)
            training_args2['sigma_small'] = True
            training_args2['diffusion_steps'] = 200  # 500  # DEBUG
            temp_dict = model_and_diffusion_defaults()
            temp_dict.update(training_args2)
            _, diffusion = create_model_and_diffusion(**temp_dict)
            config.input_emb_dim = model_args.n_embd
            config.train_diff_steps = training_args2['diffusion_steps']
            model = Classifier_Consistency(config=config, diffusion=diffusion,)
            path_save = '/u/scr/nlp/xlisali/predictability/diffusion_models_v7/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e_long/model750000.pt'
            embedding_weight = torch.load(path_save)['word_embedding.weight']
            model.bert.embeddings.word_embeddings.weight = embedding_weight
            model.bert.embeddings.word_embeddings.weight.requires_grad = False
        else:
            config.vocab_size = len(tokenizer)
            model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-gen-pos']:
        tokenize_function, pad_function = prepare_e2e_pos(raw_datasets, tokenizer, exp_vocab, model_args)
    elif model_args.experiment in ['e2e-tgt-gen-length']:
        tokenize_function, pad_function = prepare_e2e_length(raw_datasets, tokenizer, exp_vocab, model_args)
    elif model_args.experiment in ['e2e-tgt-tree', 'e2e-tgt-gen-tree', 'e2e-tgt-gen-spans']:
        tokenize_function, pad_function = prepare_e2e_tree(raw_datasets, tokenizer, parser, exp_vocab, model_args)
    elif model_args.experiment in ['e2e-back', 'e2e-back-gen']:
        tokenize_function, pad_function = prepare_e2e_back(raw_datasets, tokenizer, model_args)
    elif model_args.experiment in ['roc', 'simple-wiki', 'e2e-tgt']:
        if model_args.task not in ['data_teacher', 'finetune']:
            def tokenize_function(examples):
                vocab_dict = raw_datasets.vocab
                if model_args.task == 'classifier':
                    input_ids = [
                        [0] +
                        [vocab_dict.get(x, vocab_dict['UNK']) for x in seq1 + seq2 + seqmid] +
                        [1]
                        for (seq1, seq2, seqmid) in zip(
                            examples['left_text'], examples['right_text'], examples['mid_text']
                        )
                    ]

                    type_ids = [
                        [0] +
                        [0] * (len(seq1) + len(seq2)) +
                        [1] * len(seqmid) +
                        [1]
                        for (seq1, seq2, seqmid) in zip(
                            examples['left_text'],
                            examples['right_text'],
                            examples['mid_text']
                        )
                    ]

                    labels = examples['label']
                    result_dict = {'input_ids': input_ids, 'type_ids': type_ids, 'labels': labels}
                else:
                    input_ids = [
                        [0] +
                        [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] +
                        [1]
                        for seq in examples['text']
                    ]
                    result_dict = {'input_ids': input_ids}
                return result_dict

            if model_args.padding_mode == 'block':
                pad_function = lambda sample: group_texts(sample, block_size=64)
            else:
                def pad_function(group_lst):
                    vocab_dict = raw_datasets.vocab
                    max_length = 64
                    if model_args.task == 'classifier':
                        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'],
                                                                       max_length)
                        group_lst['type_ids'] = _collate_batch_helper(group_lst['type_ids'], 2,
                                                                       max_length)
                        group_lst["labels"] = group_lst["labels"]
                    else:
                        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
                        group_lst["labels"] = group_lst["input_ids"].copy()

                    return group_lst
        else:
            tokenize_function = lambda examples: tokenizer([
                tokenizer.bos_token + x + tokenizer.eos_token for x in examples[text_column_name]
            ])

            def pad_function(group_lst):
                max_length = 100
                group_lst['input_ids'], group_lst['attention_mask'] = _collate_batch_helper(
                    group_lst['input_ids'],
                    tokenizer.pad_token_id,
                    max_length,
                    return_mask=True,
                    pad_mask_id=0
                )
                group_lst["labels"] = group_lst["input_ids"].copy()
                return group_lst
    else:
        tokenize_function = lambda examples: tokenizer(examples[text_column_name])
        block_size = min(data_args.block_size, tokenizer.model_max_length)
        pad_function = lambda examples: group_texts(examples, block_size=block_size)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"padding",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()
    trainer.evaluate()


if __name__ == "__main__":
    main()

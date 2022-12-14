from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from controlable_generation.experiments.common import model_and_diffusion


class PartOfSpeechBase:
    def __init__(self, model_name_or_path: str):
        self.special_tokens = ['START', 'END', 'UNK', 'PAD']
        self.parts_of_speech = [
            'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN',
            'VERB', 'ADP', 'AUX', 'CCONJ', 'DET',
            'NUM', 'PART', 'PRON', 'SCONJ', 'PUNCT',
            'SYM', 'X'
        ]
        self.vocab = {e: i for i, e in enumerate(self.parts_of_speech + self.special_tokens)}

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})


class PartOfSpeech(PartOfSpeechBase):
    def __init__(self, model_name_or_path: str, init_emb: str):
        super(PartOfSpeech, self).__init__(model_name_or_path=model_name_or_path)
        self.tokenizer.add_tokens(list(self.vocab.keys()))

        _, diffusion = model_and_diffusion(init_emb)


class PartOfSpeechGen(PartOfSpeechBase):
    def __init__(self, model_name_or_path: str):
        super(PartOfSpeechGen, self).__init__(model_name_or_path=model_name_or_path)
        self.tokenizer.add_tokens(list(self.vocab.keys()))

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

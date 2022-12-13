import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import NextSentencePredictorOutput
from transformers.models.bert.modeling_bert import BertOnlyNSPHead


class Classifier_Consistency(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, diffusion=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.bert.embeddings.word_embeddings = nn.Embedding(config.vocab_size, config.input_emb_dim, )
        self.cls = BertOnlyNSPHead(config)
        self.up_proj = nn.Sequential(nn.Linear(config.input_emb_dim, config.input_emb_dim * 4), nn.Tanh(),
                                     nn.Linear(config.input_emb_dim * 4, config.hidden_size))
        print(diffusion)
        self.diffusion = diffusion
        if diffusion is not None:
            self.train_diff_steps = config.train_diff_steps
        else:
            self.train_diff_steps = 200
        self.time_embeddings = nn.Embedding(self.train_diff_steps + 1, config.hidden_size)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def set_input_embeddings(self, new_embeddings):
        self.bert.embeddings.word_embeddings = new_embeddings

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            context_ids=None,
            type_ids=None,
            mid_ids=None,
            input_embs=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            t=200,
            t_aware=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            assert context_ids is not None and (mid_ids is not None or input_embs is not None)
            context_embs = self.bert.embeddings.word_embeddings(context_ids)
            context_type_ids = torch.full_like(context_ids, 0)
        else:
            assert type_ids is not None
            context_input_embs = self.bert.embeddings.word_embeddings(input_ids)
            context_input_type_ids = type_ids
            input_embs = context_input_embs.clone()

        if self.diffusion is not None:
            if self.train_diff_steps > 0:
                t = torch.randint(-1, self.train_diff_steps, (input_embs.shape[0],)).to(input_embs.device)
                t_mask = (t >= 0)
                input_embs_rand = self.diffusion.q_sample(input_embs, t)
                input_embs[t_mask] = input_embs_rand[t_mask]
                t[~t_mask] = self.train_diff_steps
                time_emb = self.time_embeddings(t).unsqueeze(1)

        if self.diffusion is None and t is not None:
            t = torch.LongTensor([t]).expand(input_embs.size(0)).to(self.device)
            time_emb = self.time_embeddings(t).unsqueeze(1)

        context_input_embs[context_input_type_ids == 1] = input_embs[context_input_type_ids == 1]
        context_input_embs = self.up_proj(context_input_embs)

        input_embs = context_input_embs  # torch.cat([context_embs, context_input_embs], dim=1)
        if t_aware:
            input_embs = torch.cat([time_emb, input_embs], dim=1)
            t_type_ids = torch.LongTensor([0]).unsqueeze(0).expand(input_embs.shape[0], -1).to(self.device)
            token_type_ids = torch.cat([t_type_ids, context_input_type_ids], dim=1)

        attention_mask = (token_type_ids != 2)

        transformer_outputs = self.bert(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if t_aware and past_key_values is None:
            hidden_states = transformer_outputs[0][:, 1:, ]
        else:
            hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        pooled_output = transformer_outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
        )

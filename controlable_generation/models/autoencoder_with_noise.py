from typing import Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, \
    GPT2LMHeadModel, BertForMaskedLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AutoEncoderWithNoise(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, config2=None):
        super().__init__(config)
        self.bert = BertModel(config)
        if hasattr(config, 'reduced_emb'):
            self.mlp_dim = config.reduced_emb
        else:
            self.mlp_dim = 8

        if hasattr(config, 'sigma'):
            self.sigma = config.sigma
        else:
            self.sigma = 1.
        self.down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size // 2), nn.Tanh(),
                                       nn.Linear(config.hidden_size // 2, self.mlp_dim))
        self.up_proj = nn.Sequential(nn.Linear(self.mlp_dim, config.hidden_size // 2), nn.Tanh(),
                                     nn.Linear(config.hidden_size // 2, config.hidden_size))

        if hasattr(config, 'rounding_mode'):
            self.rounding_mode = config.rounding_mode
        else:
            self.rounding_mode = 'gpt2'

        if self.rounding_mode == 'gpt2':
            config2.vocab_size = config.vocab_size
            self.decoder = GPT2LMHeadModel(config2)
        elif self.rounding_mode == 'bert':
            self.decoder = BertForMaskedLM(config)
        elif self.rounding_mode == 'conv':
            raise NotImplementedError
        elif self.rounding_mode == 'mlp':
            self.decoder = nn.Sequential(nn.Linear(self.mlp_dim, config.hidden_size // 2), nn.Tanh(),
                                         nn.Linear(config.hidden_size // 2, config.hidden_size))
        else:
            assert False, 'invalid rounding_mode'
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids=None,
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
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.bert(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        down_proj = self.down_proj(hidden_states)
        down_proj_norm = torch.norm(down_proj, dim=-1)  # bsz, seqlen
        clamped_norm = torch.clamp(down_proj_norm, min=1)  # bsz, seqlen
        clamped_down_proj = down_proj / clamped_norm.unsqueeze(-1)
        gaussian_noise = torch.randn(clamped_down_proj.shape).to(clamped_down_proj.device) * self.sigma
        noised_z = clamped_down_proj + gaussian_noise

        if self.rounding_mode == 'gpt2':
            embs2 = self.up_proj(noised_z)
            decoder_outputs = self.decoder(input_ids, encoder_hidden_states=embs2)
            lm_logits = decoder_outputs.logits[:, :-1].contiguous()
            labels = labels[..., 1:].contiguous()
            # version 1. concatenate these at the beginning to attend to them all at once.
        elif self.rounding_mode == 'gpt2_v2':
            # version 2. concatenate these at each token position, one by one.
            embs2 = self.up_proj(noised_z)
            input_ids_embs = self.decoder.embeddings(input_ids)
            concat_embs = embs2 + input_ids_embs  # bsz, seqlen, dim
            decoder_outputs = self.decoder(inputs_embeds=concat_embs)
            lm_logits = decoder_outputs.logits[:, :-1].contiguous()
            labels = labels[..., 1:].contiguous()
        elif self.rounding_mode == 'gpt2_v3':
            # version 2. concatenate these at each token position, one by one.
            embs2 = self.up_proj(noised_z)
            input_ids_embs = self.decoder.embeddings(input_ids)
            concat_embs = torch.cat([embs2, input_ids_embs], dim=1)  # bsz, seqlen, dim
            decoder_outputs = self.decoder(inputs_embeds=concat_embs)
            lm_logits = decoder_outputs.logits[:, embs2.size(1) - 1:-1].contiguous()
        elif self.rounding_mode == 'bert':
            embs2 = self.up_proj(noised_z)
            decoder_outputs = self.decoder(inputs_embeds=embs2)
            lm_logits = decoder_outputs.logits

        elif self.rounding_mode == 'conv':
            raise NotImplementedError

        elif self.rounding_mode == 'mlp':
            embs2 = self.up_proj(noised_z)
            embs2 = self.decoder(embs2)
            lm_logits = self.lm_head(embs2)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits
            shift_labels = labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=down_proj,  # transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def half_forward(
            self,
            input_ids=None,
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
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.bert(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        down_proj = self.down_proj(hidden_states)
        down_proj_norm = torch.norm(down_proj, dim=-1)  # bsz, seqlen
        clamped_norm = torch.clamp(down_proj_norm, min=1)  # bsz, seqlen
        clamped_down_proj = down_proj / clamped_norm.unsqueeze(-1)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        lm_logits = None

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # print(lm_logits.shape)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=clamped_down_proj,  # transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


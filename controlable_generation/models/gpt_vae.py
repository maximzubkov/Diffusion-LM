from typing import Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Model
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GPT2VAE(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if hasattr(config, 'sigma'):
            self.sigma = config.sigma
        else:
            self.sigma = 1.

        if hasattr(config, 'reduced_emb'):
            self.latent_dim = config.reduced_emb
        else:
            self.latent_dim = 8

        if hasattr(config, 'mlp_dim'):
            self.mlp_dim = config.mlp_dim
        else:
            self.mlp_dim = 128

        self.q = nn.Embedding(config.vocab_size, self.latent_dim)
        self.g_theta = nn.Sequential(nn.Linear(self.latent_dim, self.mlp_dim), nn.Tanh(),
                                     nn.Linear(self.mlp_dim, config.vocab_size))

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
        bsz, seqlen = input_ids.shape

        embs = self.q(input_ids)
        z_sample = torch.randn(bsz, seqlen, self.latent_dim).to(embs.device)
        z_embed = z_sample * self.sigma + embs

        logits = self.g_theta(z_embed)
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_first = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss_first = loss_first.view(labels.shape)[:, 1:]

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        lm_logits = lm_logits[..., :-1, :].contiguous()  # bsz, seqlen-1, vocab

        embeds_q = self.q.weight
        vocab_size = embeds_q.size(0)

        # z_embed.shape # bsz * seqlen, dim
        # embeds_q # vocab, dim
        D = torch.sum(z_embed.view(-1, z_embed.size(-1)) ** 2, axis=-1).unsqueeze(1).expand(-1, vocab_size) + \
            torch.sum(embeds_q ** 2, axis=-1).unsqueeze(0).expand(bsz * seqlen, -1) \
            - 2 * torch.mm(z_embed.view(-1, z_embed.size(-1)),
                           embeds_q.transpose(0, 1)).view(bsz * seqlen, vocab_size)
        D = D.view(bsz, seqlen, vocab_size)
        D = D[:, 1:].contiguous()
        nll = -torch.logsumexp(lm_logits - D / (2.0 * self.sigma ** 2), dim=-1) \
              + torch.logsumexp(lm_logits, dim=-1)

        loss = loss_first - nll
        # print(loss_first.shape, nll.shape, loss.shape)
        loss = loss.mean()

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = lm_logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

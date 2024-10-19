import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartEncoder, BartPreTrainedModel, BartScaledWordEmbedding, \
    BartLearnedPositionalEmbedding, BartDecoderLayer


def tokenize_and_pad(string_list, max_length, pad_token_id):
    """
    Tokenize a list of strings and pad them to max_length.

    Args:
    string_list (list of str): List of input strings
    max_length (int): Maximum length to pad to
    pad_token_id (int): ID of the padding token

    Returns:
    tokens (torch.Tensor): Tensor of tokenized and padded inputs
    attention_mask (torch.Tensor): Attention mask for padded inputs
    """
    tokens = []
    for string in string_list:
        # Tokenize and truncate if necessary
        token_list = [ord(c) for c in string[:max_length]]
        # Pad if necessary
        token_list += [pad_token_id] * (max_length - len(token_list))
        tokens.append(token_list)

    tokens = torch.tensor(tokens)
    attention_mask = (tokens != pad_token_id).float()

    return tokens, attention_mask


class EncoderModel(nn.Module):
    def __init__(self, vocab_size=128, embed_size=1024, hidden_size=4816, max_length=14, num_encoder_layers=6, num_heads=8):
        super(EncoderModel, self).__init__()
        self.max_length = max_length
        self.pad_token_id = vocab_size  # Use the last token as padding token
        self.cls_token_id = vocab_size + 1  # Use vocab_size + 1 as [CLS] token

        # Embedding layer (increase vocab_size by 2 to accommodate pad token and [CLS] token)
        self.embedding = nn.Embedding(vocab_size + 2, embed_size, padding_idx=self.pad_token_id)

        # BartEncoder configuration
        bart_config = BartConfig(
            vocab_size=vocab_size + 2,  # Include pad token and [CLS] token in vocab
            d_model=embed_size,
            encoder_layers=num_encoder_layers,
            encoder_attention_heads=num_heads,
            encoder_ffn_dim=hidden_size,
            max_position_embeddings=max_length + 1,  # +1 for [CLS] token
            pad_token_id=self.pad_token_id
        )

        # BartEncoder layer
        self.encoder = BartEncoder(bart_config)

    def forward(self, tokens, attention_mask):
        batch_size, seq_length = tokens.size()

        # Add [CLS] token at the beginning of each sequence
        cls_tokens = torch.full((batch_size, 1), self.cls_token_id, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([cls_tokens, tokens[:, :self.max_length-1]], dim=1)

        # Update attention mask for [CLS] token
        cls_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([cls_mask, attention_mask[:, :self.max_length-1]], dim=1)

        # Embed tokens
        x = self.embedding(tokens)  # Shape: [batch_size, max_length, embed_size]

        # Pass through BartEncoder
        encoder_outputs = self.encoder(input_ids=None, inputs_embeds=x, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # Shape: [batch_size, max_length, embed_size]

        # Extract [CLS] token representation
        cls_representation = sequence_output[:, 0, :]  # Shape: [batch_size, embed_size]

        # Feed-forward layers
        x = torch.sigmoid(cls_representation)  # Sigmoid to get values between 0 and 1

        return x  # Shape: [batch_size, 441]


class EncoderDecoderModel(nn.Module):
    def __init__(self,
                 vocab_size=128,
                 num_of_pixels=441,
                 embed_size=1024,
                 hidden_size=4816,
                 max_length=14,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_heads=8
                 ):
        super(EncoderDecoderModel, self).__init__()
        self.max_length = max_length
        self.num_of_pixels = num_of_pixels
        self.pad_token_id = vocab_size  # Use the last token as padding token

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embed_size, padding_idx=self.pad_token_id)

        # BartConfig
        self.bart_config = BartConfig(
            vocab_size=vocab_size + 1,  # Include pad token in vocab
            d_model=embed_size,
            encoder_layers=num_encoder_layers,
            decoder_layers=num_decoder_layers,
            encoder_attention_heads=num_heads,
            decoder_attention_heads=num_heads,
            encoder_ffn_dim=hidden_size,
            decoder_ffn_dim=hidden_size,
            max_position_embeddings=max(max_length, num_of_pixels),
            pad_token_id=self.pad_token_id
        )

        # BartEncoder and BartDecoder
        self.encoder = BartEncoder(self.bart_config)
        self.decoder = BartDecoder(self.bart_config)

        # # Individual layers for each pixel
        self.pixel_layers = nn.ModuleList([nn.Linear(embed_size, 1) for _ in range(num_of_pixels)])

    def forward(self, tokens, attention_mask):
        batch_size, seq_length = tokens.size()

        # Embed tokens
        encoder_inputs_embeds = self.embedding(tokens)  # Shape: [batch_size, max_length, embed_size]

        # Pass through BartEncoder
        encoder_outputs = self.encoder(input_ids=None, inputs_embeds=encoder_inputs_embeds,
                                       attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state  # Shape: [batch_size, max_length, embed_size]

        # Prepare decoder inputs
        decoder_inputs_embeds = torch.zeros((batch_size, self.num_of_pixels, self.bart_config.d_model),
                                            device=tokens.device)

        # Pass through BartDecoder
        decoder_outputs = self.decoder(
            input_ids=None,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )
        decoder_hidden_states = decoder_outputs.last_hidden_state  # Shape: [batch_size, num_of_pixels, embed_size]

        # Apply individual layers for each pixel
        output = torch.zeros((batch_size, self.num_of_pixels), device=tokens.device)
        for i, layer in enumerate(self.pixel_layers):
            output[:, i] = layer(decoder_hidden_states[:, i, :]).squeeze(-1)

        output = torch.sigmoid(output)

        return output  # Shape: [batch_size, 441]


class BartDecoder(BartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions and cross_attn_head_mask is None:
            # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask,
                inputs_embeds.dtype,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype,
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self._use_sdpa and cross_attn_head_mask is None and not output_attentions:
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BartConfig, T5Config, T5Model
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder


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
        # x = self.f1(cls_representation)
        x = torch.sigmoid(cls_representation)  # Sigmoid to get values between 0 and 1

        return x  # Shape: [batch_size, 441]


class T5EncoderDecoderModel(nn.Module):
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
        super(T5EncoderDecoderModel, self).__init__()
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

        # Individual layers for each pixel
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

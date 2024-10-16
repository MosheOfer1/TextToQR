import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder


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


class Model(nn.Module):
    def __init__(self, vocab_size=128, embed_size=256, hidden_size=512, max_length=14, num_encoder_layers=3):
        super(Model, self).__init__()
        self.max_length = max_length
        self.pad_token_id = vocab_size  # Use the last token as padding token

        # Embedding layer (increase vocab_size by 1 to accommodate pad token)
        self.embedding = nn.Embedding(vocab_size + 1, embed_size, padding_idx=self.pad_token_id)

        # BartEncoder configuration
        bart_config = BartConfig(
            vocab_size=vocab_size + 1,  # Include pad token in vocab
            d_model=embed_size,
            encoder_layers=num_encoder_layers,
            encoder_attention_heads=8,
            encoder_ffn_dim=hidden_size,
            max_position_embeddings=max_length,
            pad_token_id=self.pad_token_id
        )

        # BartEncoder layer
        self.encoder = BartEncoder(bart_config)

        # Attention layer for sequence representation
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=1)

        self.f1 = nn.Linear(embed_size, 441)  # 21x21 = 441

    def forward(self, tokens, attention_mask):
        batch_size = tokens.size(0)

        # Embed tokens
        x = self.embedding(tokens)  # Shape: [batch_size, max_length, embed_size]

        # Pass through BartEncoder
        encoder_outputs = self.encoder(input_ids=None, inputs_embeds=x, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # Shape: [batch_size, max_length, embed_size]

        # Apply attention to get sequence representation
        sequence_output = sequence_output.transpose(0, 1)  # Shape: [max_length, batch_size, embed_size]
        query = self.f1.weight.mean(dim=0, keepdim=True).unsqueeze(1).expand(-1, batch_size, -1)
        context, _ = self.attention(query, sequence_output, sequence_output, key_padding_mask=attention_mask.eq(0))
        sequence_repr = context.squeeze(0)  # Shape: [batch_size, embed_size]

        # Feed-forward layers
        x = self.f1(sequence_repr)
        x = torch.sigmoid(x)  # Sigmoid to get values between 0 and 1

        return x  # Shape: [batch_size, 441]



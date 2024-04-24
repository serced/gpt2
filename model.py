# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt


# %%
def get_positional_encoding(max_seq_len, d_model):
    pos_enc = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return torch.Tensor(pos_enc)


# %%
# Now we can tackle the main part of the model: the decoder block.


# The main components are masked self-attention and a standard neural network.
class MultiHeadMaskedSelfAttention(nn.Module):
    """
    When we create a MultiHeadMaskedSelfAttention module,
    we need to pass the arguments in the `__init__`. These are:

    embedding_size: the vector dimensions
    number_of_heads: the number of attention heads, i.e., the "multi" number
    bias and dropout can be ignored for now
    """

    def __init__(self, embedding_size, number_of_heads, bias, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.bias = bias
        self.number_of_heads = number_of_heads
        assert (
            embedding_size % number_of_heads == 0
        ), "Embedding dimension has to be divisible by the number of heads"

        # A simple linear transformation that projects the input into three different spaces
        self.queries_projection = nn.Linear(embedding_size, embedding_size)
        self.keys_projection = nn.Linear(embedding_size, embedding_size)
        self.values_projection = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        # See note 1
        # Input (x) shape is (batch_size, sequence_length, embedding_size)
        batch_size, sequence_length, embedding_size = x.size()
        queries = self.queries_projection(x)
        keys = self.keys_projection(x)
        values = self.values_projection(x)

        # See note 2
        head_size = embedding_size // self.number_of_heads
        queries = queries.view(
            batch_size, sequence_length, self.number_of_heads, head_size
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, sequence_length, self.number_of_heads, head_size
        ).transpose(1, 2)
        values = values.view(
            batch_size, sequence_length, self.number_of_heads, head_size
        ).transpose(1, 2)

        # See note 3
        mask = torch.triu(
            torch.ones((sequence_length, sequence_length), device=x.device),
            diagonal=1,
        )  # Strangely diagonal=1 means the diagonal is 0
        autoregressive_attention_mask = mask.masked_fill(mask == 1, float("-inf"))

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) * (
            1.0 / np.sqrt(keys.size(-1))
        )
        masked_attention_scores = attention_scores + autoregressive_attention_mask

        # Compute self-attention values
        masked_self_attention = F.softmax(masked_attention_scores, dim=-1)
        masked_self_attention = self.dropout(masked_self_attention)

        # Shape: (batch_size, number_of_heads, sequence_length, head_size)
        attended_values = torch.matmul(masked_self_attention, values)

        # Concatenate the heads
        # Shape: (batch_size, sequence_length, embedding_size)
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, embedding_size)
        )

        # Pass through a linear layer to get the final output
        output = self.output_projection(attended_values)

        return output


# %%
class MultiLayerPerceptron(nn.Module):
    """
    input_size: the dimensions of the input
    hidden_size: how wide the neural network layer should be
    output_size: the dimensions you want the output to have
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_linear_layer = nn.Linear(input_size, hidden_size)
        self.output_linear_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_linear_layer(x)
        x = F.gelu(x)
        x = self.output_linear_layer(x)

        return x


# %%
class TransformerDecoderBlock(nn.Module):
    """
    Take a look at the architecture image and you will notice that we now define
    all required parts for the transformer block in our `__init__` function and
    then use them in our `forward pass` on the input x.


    embedding_size: the vector dimensions per character
    n_attention_heads: the number of attention heads
    """

    def __init__(self, embedding_size, n_attention_heads, dropout=0.1):
        super().__init__()
        self.input_layer_normalization = nn.LayerNorm(embedding_size)

        # Here we use our own MultiHeadMaskedSelfAttention class
        self.masked_attention = MultiHeadMaskedSelfAttention(
            embedding_size, n_attention_heads, bias=True, dropout=dropout
        )
        self.post_attention_dropout = nn.Dropout(dropout)

        self.pre_mlp_layer_normalization = nn.LayerNorm(embedding_size)
        # Here we use our own MultiLayerPerceptron class
        self.multi_layer_perceptron = MultiLayerPerceptron(
            input_size=embedding_size,
            hidden_size=embedding_size * 4,
            output_size=embedding_size,
        )
        self.post_mlp_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape is (batch_size, sequence_length, embedding_dimension)
        # First residual connection wrapper
        residual = x.clone()
        x = self.input_layer_normalization(x)
        x = self.masked_attention(x)
        x = self.post_attention_dropout(x)
        x = x + residual

        x = self.pre_mlp_layer_normalization(x)

        # Second residual connection wrapper
        residual = x.clone()
        x = self.multi_layer_perceptron(x)
        x = self.post_mlp_dropout(x)
        x = x + residual

        return x


# %%
class GenerativePretrainedTransformer(nn.Module):
    """
    embedding_dimension: the number of vector dimensions that represent a single character
    sequence_length: how many characters the model can process at most
    n_transformer_blocks: the number of transformer decoder blocks used
    n_attention_heads: the number of attention heads in each decoder block
    vocabulary_size: the number of unique characters in our dataset
    """

    def __init__(
        self,
        embedding_dimension,
        maximum_sequence_length,
        n_transformer_blocks,
        n_attention_heads,
        vocabulary_size,
        dropout=0.1,
    ):
        super().__init__()
        self.maximum_sequence_length = maximum_sequence_length
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # A matrix mapping characters to vectors and the static positional encoding matrix
        self.character_to_embedding_map = nn.Embedding(
            vocabulary_size, embedding_dimension
        )
        self.positional_encoding = get_positional_encoding(
            maximum_sequence_length, embedding_dimension
        ).to(device)

        # Define the transformer decoder blocks used
        self.transformer_decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embedding_dimension, n_attention_heads, dropout=dropout
                )
                for _ in range(n_transformer_blocks)
            ]
        )

        # The normalization layer and the layer used for predicting which character comes next
        self.layer_normalization = nn.LayerNorm(embedding_dimension)
        self.character_prediction_layer = nn.Linear(
            embedding_dimension, vocabulary_size, bias=False
        )

    def forward(self, batch_of_character_sequences):
        # Input shape: (batch_size, sequence_length)
        x = self.character_to_embedding_map(batch_of_character_sequences)
        character_positions_in_sequence = torch.arange(x.size(1))

        x = x + self.positional_encoding[character_positions_in_sequence]

        # Pass through all Transformer Decoder Blocks
        # x shape is (batch_size, sequence_length, embedding_dimension)
        for decoder_block in self.transformer_decoder_blocks:
            x = decoder_block(x)

        x = self.layer_normalization(x)
        # These are called logits because you need to normalize them with F.softmax
        # when actually generating text (see the `generate` function for this)
        output_logits = self.character_prediction_layer(x)
        return output_logits

    @torch.no_grad()
    def generate(self, input_text_indices, max_length=300, temperature=1.0, top_k=None):
        """
        The generate function is used after training and we can use it to
        predict the next characters from any input sequence.
        """
        input_text_indices = input_text_indices.unsqueeze(0)
        current_sequence_length = input_text_indices.size(1)

        for _ in range(max_length):
            # Check that the context is not too long, otherwise cut it
            if input_text_indices.size(1) <= MAXIMUM_SEQUENCE_LENGTH:
                input_text_indices = input_text_indices[:, -MAXIMUM_SEQUENCE_LENGTH:]

            # Pass through the model
            logits = self.forward(input_text_indices)
            # We only need the logits for the last character
            next_character_logits = logits[0, -1, :] / temperature
            # Apply top-k sampling if needed
            if top_k is not None:
                top_k_characters, top_k_indices = torch.topk(next_character_logits)
                # Set all logits to -infinity that are not in the top-k
                # Effectively setting the probability to 0 after the softmax
                next_character_logits[~top_k_indices] = float("-inf")

            # Transform the model prediction into a probability distribution and sample from it
            next_character_probabilities = F.softmax(next_character_logits, dim=-1)
            next_character = torch.multinomial(next_character_probabilities, 1)
            # Append to the sequence
            input_text_indices = torch.cat(
                [input_text_indices, next_character.unsqueeze(0)], dim=1
            )
            current_sequence_length += 1

        return input_text_indices


# %%

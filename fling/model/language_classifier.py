import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from fling.utils.registry_utils import MODEL_REGISTRY
from .resnet import FedRodHead


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


@MODEL_REGISTRY('transformer_classifier')
class TransformerClassifier(nn.Module):

    def __init__(self, vocab_size: int, hidden_dim: int = 200, n_head: int = 2, ffn_dim: int = 2,
                 n_layers: int = 4, class_number: int = 5, dropout: float = 0.1, fedrod_head: bool = False):
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = TransformerEncoderLayer(hidden_dim, n_head, ffn_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.d_model = hidden_dim
        if not fedrod_head:
            self.classifier = nn.Linear(hidden_dim, class_number)
        else:
            self.classifier = FedRodHead(hidden_dim, class_number)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            - src: Tensor, shape ``[T, B]``

        Returns:
            - output: Tensor of shape ``[B, n_class]``
        """
        src = torch.transpose(src, 0, 1)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)[-1, :, :]

        output = self.classifier(output)

        return output

"""
For the original paper of Vision Transformer (ViT), please refer to: https://arxiv.org/pdf/2010.11929.pdf.
This implementation follows: https://github.com/lucidrains/vit-pytorch.
"""
from typing import Union, Tuple
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from fling.utils.registry_utils import MODEL_REGISTRY


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class FeedForward(nn.Module):
    """
    Overview:
        The implementation of Feed-Forward-Network (FFN). This module contains to linear layers and corresponding \
        layer-norm layers and dropout layers. The activation function is set to GELU by default. The input and output \
        shape of tensors is the same for this module.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        """
        Overview:
            Initialize the FeedForward module using the given arguments.
        Arguments:
            - dim: The input dimension for the input tensor, such as 32.
            - hidden_dim: The dimension for the hidden-states in this module, such as 128.
            - dropout: The drop-rate of the dropout layers in this module. For the default case, this is set to 0, \
                which means that no dropout will be applied.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            The computation graph of FeedForward module.
        Arguments:
            - x: The input tensor.
        Returns:
            - output: The output tensor, whose shape is the same as that of the input x.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Overview:
        The implementation of self-attention module in vision transformer. For this module, the shape of input \
        tensor and output tensor is the same.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        """
        Overview:
            Initialize the Attention module using the given arguments.
        Arguments:
            - dim: The input dimension for the input tensor, such as 32.
            - heads: The number of heads in self-attention module, such as 8.
            - dim_head: The dimension for attention for each head, such as 64.
            - dropout: The drop-rate of the dropout layers in this module. For the default case, this is set to 0, \
                which means that no dropout will be applied.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            The computation graph of Attention module.
        Arguments:
            - x: The input tensor.
        Returns:
            - output: The output tensor, whose shape is the same as that of the input x.
        """
        x = self.norm(x)

        # Calculate qkv.
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Add a new axis: head.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Soft-max function.
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Overview:
        The implementation of transformer module in vision transformer. For this module, the shape of input \
        tensor and output tensor is the same.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        """
        Overview:
            Initialize the Transformer module using the given arguments.
        Arguments:
            - dim: The input dimension for the input tensor, such as 32.
            - depth: The depths of this transformer block. For example, ``depth=4`` means that this module have 4 \
                attention blocks and 4 FFNs.
            - heads: The number of heads in self-attention module, such as 8.
            - dim_head: The dimension for attention for each head, such as 64.
            - dropout: The drop-rate of the dropout layers in this module. For the default case, this is set to 0, \
                which means that no dropout will be applied.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            The computation graph of Transformer module.
        Arguments:
            - x: The input tensor.
        Returns:
            - output: The output tensor, whose shape is the same as that of the input x.
        """
        for attn, ff in self.layers:
            # Forward computation through each attention module and FNN.
            # Residual link.
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


@MODEL_REGISTRY.register('vit')
class ViT(nn.Module):
    """
    Overview:
        The implementation of Vision Transformer (ViT). For the original paper, you can refer to: \
        https://arxiv.org/pdf/2010.11929.pdf. Essentially, it is a classification model, whose input is an image and \
        the output is a logit for all classes.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
        self,
        *,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        class_number: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = 'cls',
        input_channel: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.
    ):
        """
        Overview:
            Initialize the ViT module using the given arguments.
        Arguments:
            - image_size: The input image size. It can be either a number like: 224, or a tuple like: (224, 224).
            - patch_size: The path size of image. It can be either a number like: 16, or a tuple like: (16, 16).
            - class_number: The number of classes for the classification task, such as 10.
            - dim: The input dimension for the input tensor, such as 32.
            - depth: The depths of this transformer block. For example, ``depth=4`` means that this module have 4 \
                attention blocks and 4 FFNs.
            - heads: The number of heads in self-attention module, such as 8.
            - mlp_dim: The dimension of feed-forward network, such as 128.
            - pool: What kind of embedding should be used for classification. "cls" means that the first token will \
                be used, "mean" means that the mean result of all tokens will be used.
            - input_channel: The number of channels for input images.
            - dim_head: The dimension for attention for each head, such as 64.
            - dropout: The drop-rate of the dropout layers in the transformer module. For the default case, this is \
                set to 0, which means that no dropout will be applied.
            - emb_dropout: The drop-rate of the dropout layers in the embedding module. For the default case, this is \
                set to 0, which means that no dropout will be applied.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = input_channel * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Embedding layers.
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Position encoding.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # CLS token.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer layers.
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # The classification head.
        self.mlp_head = nn.Linear(dim, class_number)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            The computation graph of ViT module.
        Arguments:
            - img: The input image.
        Returns:
            - logit: The output logit, whose dimension equals to ``class_number``.
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add the cls token to all images in this batch.
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position encoding.
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    # Test function.
    # Generate fake data.
    x = torch.randn(1, 3, 128, 128)

    # Construct the model.
    model = MODEL_REGISTRY.build(
        "vit", image_size=128, patch_size=16, class_number=10, dim=64, depth=4, heads=4, mlp_dim=128
    )

    y = model(x)

    # Test the output shape.
    assert y.shape == torch.Size([1, 10])

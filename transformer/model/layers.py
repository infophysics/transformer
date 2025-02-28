import numpy as np
import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    The token embedding takes a torch tensor of size 
    (batch, vocab_size) => (batch, vocab_size, d_model),
    where d_model is the number of features in the embedding.
    The embedding features are learned, unlike the positional
    encoding.
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.config = config
        self.parse_config()
    
    def parse_config(self):
        self.d_model = self.config["d_model"]
        self.vocab_size = self.config["vocab_size"]
        self.scale = math.sqrt(self.d_model)
        
        """Construct embedding layers"""
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.d_model
        )
    
    def forward(self, data):
        return self.scale * self.embedding(data)


class PositionalEncoding(nn.Module):
    """
    The positional encoding follows the standard procedure
    of using sin/cos to generate continuous and cyclic position
    embedding vectors. One alteration to the original Transformer paper
    is the use of the log transform of the denominator term:

        div_term = exp[-log(10,000)/d_model]

    which is computationally more stable.
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.config = config
        self.parse_config()
    
    def parse_config(self):
        self.d_model = self.config["d_model"]
        self.seq_len = self.config["seq_len"]
        
        """Create the encoding"""
        pos_encoding = torch.zeros(self.seq_len, self.d_model)
        positions = torch.arange(
            0, self.seq_len,
            dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, self.d_model, 2
        ).float() * (-math.log(10000.0) / self.d_model))
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        pos_encoding.unsqueeze(0)

        """Register this encoding tensor to the buffer"""
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, data):
        return (self.pos_encoding[:, data.shape[1], :]).requires_grad_(False)



class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention which consists
    of four weight matrices, W_Q, W_K, W_V and W_O.
    The resulting Q, K, V, A and O matrices are returned.
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.config = config
        self.forward_views = {}
        self.forward_view_map = {}
        self.parse_config()
        self.register_forward_hooks()
    
    def parse_config(self):
        self.d_model = self.config["d_model"]
        self.num_heads = self.config["num_heads"]
        self.dropout = nn.Dropout(self.config["dropout"])
        self.d_k = self.d_model // self.num_heads
        """Check that the dimension of the embedding is divisible by the number of heads"""
        assert self.d_model % self.num_heads == 0, "d_model is not divisible by num_heads!"

        """Construct the weight matrices"""
        self.layers = nn.ModuleDict()
        self.layers['W_Q'] = nn.Linear(self.d_model, self.d_model, bias=False)
        self.layers['W_K'] = nn.Linear(self.d_model, self.d_model, bias=False)
        self.layers['W_V'] = nn.Linear(self.d_model, self.d_model, bias=False)
        self.layers['W_O'] = nn.Linear(self.d_model, self.d_model, bias=False)

    
    def forward_hook(self, m, i, o):
        """
        A forward hook for a particular module.
        It assigns the output to the views dictionary.
        """
        self.forward_views[self.forward_view_map[m]] = o

    def register_forward_hooks(self):
        """
        This function registers all forward hooks for the modules
        in ModuleDict.
        """
        for name, module in self._modules.items():
            if isinstance(module, nn.ModuleDict):
                for name, layer in module.items():
                    self.forward_view_map[layer] = name
                    layer.register_forward_hook(self.forward_hook)

    @staticmethod
    def attention(Q, K, V, mask):
        """Compute QK^T"""
        d_k = Q.shape[-1]
        A = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        """Apply the mask if there is one"""
        if mask is not None:
            A.masked_fill_(mask == 0, -1e9)
        A = A.softmax(dim=-1)
        return A, (A @ V)
    
    def Q(self, q):
        return self.layers['W_Q'](q)
    
    def K(self, k):
        return self.layers['W_K'](k)
    
    def V(self, v):
        return self.layers['W_V'](v)
    
    def QK(self, q, k):
        return self.layers['W_Q'](q) @ self.layers['W_K'](k).transpose(-2, -1)
    
    def OV(self, v):
        return self.layers['W_O'](self.layers['W_V'](v))

    def forward(self, q, k, v, mask):
        """Find the Q, K, V, QK^T matrices"""
        Q = self.layers['W_Q'](q)
        K = self.layers['W_K'](k)
        V = self.layers['W_V'](v)

        """Generate separate attention head inputs"""
        Q = Q.view(
            Q.shape[0], Q.shape[1],
            self.num_heads, self.d_k
        ).transpose(1, 2)
        K = K.view(
            K.shape[0], K.shape[1],
            self.num_heads, self.d_k
        ).transpose(1, 2)
        V = V.view(
            V.shape[0], V.shape[1],
            self.num_heads, self.d_k
        ).transpose(1, 2)
        
        """Compute the Attention Scores"""
        A, O = self.attention(Q, K, V, mask)

        """Combine the heads back together"""
        O = O.transpose(1, 2).contiguous().view(
            O.shape[0], -1, self.d_model
        )

        """Apply the W_O weights and return"""
        return self.layers['W_O'](O)


class FeedForward(nn.Module):
    """
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.config = config
        self.parse_config()
    
    def parse_config(self):
        self.d_model = self.config["d_model"]
        self.d_ff = self.config["d_ff"]
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(self.config["dropout"])
    
    def forward(self, data):
        return self.linear2(self.dropout(nn.functional.leaky_relu(self.linear1(data), 0.1)))


class LayerNormalization(nn.Module):
    """
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.config = config
        self.parse_config()

    def parse_config(self):
        self.num_features = self.config["num_features"]
        self.eps = self.config["eps"]
        self.alpha = nn.Parameter(torch.ones(self.num_features))
        self.beta = nn.Parameter(torch.ones(self.num_features))
    
    def forward(self, data):
        means = data.mean(dim=-1, keepdim=True)
        stds = data.std(dim=-1, keepdim=True)
        return self.alpha * (data - means) / (stds + self.eps) + self.beta


class ProjectionLayer(nn.Module):
    """
    """
    def __init__(
        self,
        config: dict
    ):
        super().__init__()
        self.config = config
        self.parse_config()

    def parse_config(self):
        self.d_model = self.config["d_model"]
        self.vocab_size = self.config["vocab_size"]
        self.projection = nn.Linear(self.d_model, self.vocab_size)
    
    def forward(self, data):
        return self.projection(data)

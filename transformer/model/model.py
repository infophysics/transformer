import numpy as np
import torch
import torch.nn as nn
import math
from pathlib import Path

from transformer.model.layers import (
    TokenEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    LayerNormalization,
    ProjectionLayer
)


class EncoderBlock(nn.Module):
    """
    An encoder block consists of a MultiHeadAttention with a 
    skip connection and then a layer normalization, followed by 
    a simple MLP also with a skip connection and normalization.
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.config = config
        self.parse_config()
    
    def parse_config(self):
        self.multi_head_attention = MultiHeadAttention({
            "d_model": self.config["d_model"],
            "num_heads": self.config["num_heads"],
            "dropout": self.config["dropout"]
        })
        self.multi_head_normalization = LayerNormalization({
            "num_features": self.config["d_model"],
            "eps": self.config["norm_eps"]
        })
        self.feed_forward = FeedForward({
            "d_model": self.config["d_model"],
            "d_ff": self.config["d_ff"],
            "dropout": self.config["dropout"]
        })
        self.feed_forward_normalization = LayerNormalization({
            "num_features": self.config["d_model"],
            "eps": self.config["norm_eps"]
        })

    def forward(self, X, mask):
        multi_head_output = self.multi_head_attention(X, X, X, mask)
        normed_multi_head_output = self.multi_head_normalization(multi_head_output + X)
        feed_forward_output = self.feed_forward(normed_multi_head_output)
        normed_feed_forward_output = self.feed_forward_normalization(
            feed_forward_output + normed_multi_head_output
        )
        return normed_feed_forward_output


class Encoder(nn.Module):
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
        self.num_layers = self.config["num_layers"]
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock({
                "d_model": self.config["d_model"],
                "num_heads": self.config["num_heads"],
                "dropout": self.config["dropout"],
                "norm_eps": self.config["norm_eps"],
                "d_ff": self.config["d_ff"]
            })
            for _ in range(self.num_layers)
        ])
    
    def forward(self, X, mask):
        for ii, layer in enumerate(self.encoder_blocks):
            X = layer(X, mask)
        return X


class DecoderBlock(nn.Module):
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
        self.self_attention = MultiHeadAttention({
            "d_model": self.config["d_model"],
            "num_heads": self.config["num_heads"],
            "dropout": self.config["dropout"]
        })
        self.self_normalization = LayerNormalization({
            "num_features": self.config["d_model"],
            "eps": self.config["norm_eps"]
        })
        self.cross_attention = MultiHeadAttention({
            "d_model": self.config["d_model"],
            "num_heads": self.config["num_heads"],
            "dropout": self.config["dropout"]
        })
        self.cross_normalization = LayerNormalization({
            "num_features": self.config["d_model"],
            "eps": self.config["norm_eps"]
        })
        self.feed_forward = FeedForward({
            "d_model": self.config["d_model"],
            "d_ff": self.config["d_ff"],
            "dropout": self.config["dropout"]
        })
        self.feed_forward_normalization = LayerNormalization({
            "num_features": self.config["d_model"],
            "eps": self.config["norm_eps"]
        })
    
    def forward(self, Y, O, X_mask, Y_mask):
        """Compute the self attention of Y"""
        self_output = self.self_attention(Y, Y, Y, Y_mask)
        normed_self_output = self.self_normalization(self_output + Y)
        """Compute the cross attention between X and Y"""
        cross_output = self.cross_attention(normed_self_output, O, O, X_mask)
        normed_cross_output = self.cross_normalization(cross_output + normed_self_output)
        """Feed to the feed forward"""
        feed_forward_output = self.feed_forward(normed_cross_output)
        normed_feed_forward_output = self.feed_forward_normalization(
            feed_forward_output + normed_cross_output
        )
        return normed_feed_forward_output


class Decoder(nn.Module):
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
        self.num_layers = self.config["num_layers"]
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock({
                "d_model": self.config["d_model"],
                "num_heads": self.config["num_heads"],
                "dropout": self.config["dropout"],
                "norm_eps": self.config["norm_eps"],
                "d_ff": self.config["d_ff"]
            })
            for _ in range(self.num_layers)
        ])
    
    def forward(self, Y, O, X_mask, Y_mask):
        for ii, layer in enumerate(self.decoder_blocks):
            Y = layer(Y, O, X_mask, Y_mask)
        return Y


class Transformer(nn.Module):
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
        self.source_embedding = TokenEmbedding({
            "d_model": self.config["d_model"],
            "vocab_size": self.config["vocab_size"]
        })
        self.target_embedding = TokenEmbedding({
            "d_model": self.config["d_model"],
            "vocab_size": self.config["vocab_size"]
        })
        self.source_pos_encoding = PositionalEncoding({
            "d_model": self.config["d_model"],
            "seq_len": self.config["seq_len"]
        })
        self.target_pos_encoding = PositionalEncoding({
            "d_model": self.config["d_model"],
            "seq_len": self.config["seq_len"]
        })
        self.encoder = Encoder({
            "num_layers": self.config["num_layers"],
            "d_model": self.config["d_model"],
            "num_heads": self.config["num_heads"],
            "dropout": self.config["dropout"],
            "norm_eps": self.config["norm_eps"],
            "d_ff": self.config["d_ff"]
        })
        self.decoder = Decoder({
            "num_layers": self.config["num_layers"],
            "d_model": self.config["d_model"],
            "num_heads": self.config["num_heads"],
            "dropout": self.config["dropout"],
            "norm_eps": self.config["norm_eps"],
            "d_ff": self.config["d_ff"]
        })
        self.projection = ProjectionLayer({
            "d_model": self.config["d_model"],
            "vocab_size": self.config["vocab_size"]
        })
    
    def get_weights_file_path(
        self,
        model_folder,
        model_filename,
    ):
        return str(Path('.') / model_folder / model_filename)

    # Find the latest weights file in the weights folder
    def latest_weights_file_path(config):
        model_folder = f"{config['datasource']}_{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])
    
    def embed(self, data):
        data['source_embedding'] = self.source_embedding(data['source_tokens'])
        data['target_embedding'] = self.target_embedding(data['target_tokens'])
        return data

    def pos_encode(self, data):
        data['source_pos_encoding'] = self.source_pos_encoding(data['source_embedding'])
        data['target_pos_encoding'] = self.target_pos_encoding(data['target_embedding'])
        return data
    
    def encode(self, data):
        data['encoder_output'] = self.encoder(
            data['source_embedding'] + data['source_pos_encoding'],
            data['source_mask']
        )
        return data
    
    def decode(self, data):
        data['decoder_output'] = self.decoder(
            data['target_embedding'] + data['target_pos_encoding'],
            data['encoder_output'],
            data['source_mask'],
            data['target_mask']
        )
        return data

    def forward(self, data):
        data = self.embed(data)
        data = self.pos_encode(data)
        data = self.encode(data)
        data = self.decode(data)
        data['projection'] = self.projection(data)
        return data

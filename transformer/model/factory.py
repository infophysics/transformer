import torch
import torch.nn as nn

from transformer.model.model import (
    EncoderBlock,
    Encoder,
    DecoderBlock,
    Decoder,
    Transformer
)


def model_factory(config):
    return None
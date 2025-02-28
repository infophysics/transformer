import torch.nn as nn


class Loss:
    """
    """
    def __init__(
        self,
        ignore_index,
        vocab_size,
        device
    ):
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=0.1
        ).to(device)
        self.vocab_size

    def loss(
        self,
        data
    ):
        return data['projection'].view(-1, self.vocab_size), data['label'].view(-1))
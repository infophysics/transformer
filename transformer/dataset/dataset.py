import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class BilingualDataset(Dataset):
    """
    """
    def __init__(
        self,
        config: dict
    ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
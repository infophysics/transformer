import torch
import torch.nn
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class BilingualDataset(Dataset):
    """
    A dataset which downloads from the hugging face datasets for two languages,
    generates tokens using the tokenizer package and prepares tensors for
    the Transformer model.
    """
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.config = config
        self.parse_config()
        self.construct_dataset()
        self.construct_tokenizers()

    def parse_config(self):
        self.seq_len = self.config["seq_len"]
        self.source_language = self.config["source_language"]
        self.target_language = self.config["target_language"]
        self.tokenizer_file = self.config["tokenizer_file"]

    def construct_dataset(self):
        self.raw_dataset = load_dataset(
            f"{self.config['datasource']}", 
            f"{self.source_language}-{self.target_language}", 
            split='train'
        )

    def construct_tokenizers(self):
        # Build tokenizers
        self.source_tokenizer = self.get_or_build_tokenizer(self.source_language)
        self.target_tokenizer = self.get_or_build_tokenizer(self.target_language)
        self.sos_token = torch.tensor([self.target_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.target_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.target_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        # Find the maximum length of each sentence in the source and target sentence
        self.max_len_src = 0
        self.max_len_tgt = 0

        for item in self.raw_dataset:
            src_ids = self.source_tokenizer.encode(item['translation'][self.source_language]).ids
            tgt_ids = self.target_tokenizer.encode(item['translation'][self.target_language]).ids
            self.max_len_src = max(self.max_len_src, len(src_ids))
            self.max_len_tgt = max(self.max_len_tgt, len(tgt_ids))

        print(f'Max length of source sentence: {self.max_len_src}')
        print(f'Max length of target sentence: {self.max_len_tgt}')

    def get_all_sentences(
        self,
        language
    ):
        for item in self.raw_dataset:
            yield item['translation'][language]

    def get_or_build_tokenizer(
        self,
        language
    ):
        tokenizer_path = Path(self.tokenizer_file.format(language))
        if not Path.exists(tokenizer_path):
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.get_all_sentences(self.raw_dataset, language), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    def causal_mask(size):
        mask = torch.triu(
            torch.ones((1, size, size)), diagonal=1
        ).type(torch.int)
        return mask == 0

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        src_target_pair = self.raw_dataset[idx]
        source_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.self.source_tokenizer.encode(source_text).ids
        dec_input_tokens = self.self.target_tokenizer.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        source_tokens = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        target_tokens = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens,
            "source_mask": (source_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "target_mask": (
                (target_tokens != self.pad_token).unsqueeze(0).int() & 
                self.causal_mask(target_tokens.size(0))
            ), 
            "label": label,
            "source_text": source_text,
            "target_text": target_text,
        }

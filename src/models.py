"""Arsitektur Bi-GRU (student) + helper vocabulary."""
import json

import numpy as np
import torch
import torch.nn as nn

PAD_ID, UNK_ID = 0, 1


class BiGRUStudent(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_labels, emb_matrix=None,
                 freeze_emb=True, h1=128, h2=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        if emb_matrix is not None:
            self.emb.weight.data.copy_(torch.tensor(emb_matrix, dtype=torch.float32))
        self.emb.weight.requires_grad = not freeze_emb
        self.gru1 = nn.GRU(emb_dim, h1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(2 * h1, h2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * h2, 128)
        self.out = nn.Linear(128, n_labels)

    def forward(self, x):
        mask = (x != PAD_ID).unsqueeze(-1)
        o, _ = self.gru1(self.emb(x))
        o = self.drop(o)
        o, _ = self.gru2(o)
        o = self.drop(o)
        # max-pooling yang mengabaikan posisi padding
        o = o.masked_fill(~mask, -1e9).max(dim=1).values
        return self.out(torch.relu(self.fc(o)))  # LOGIT (sigmoid diterapkan di luar)


def encode(text: str, vocab: dict, max_len: int):
    ids = [vocab.get(t, UNK_ID) for t in str(text).split()][:max_len]
    if not ids:
        ids = [UNK_ID]
    return ids + [PAD_ID] * (max_len - len(ids))


def encode_many(texts, vocab, max_len):
    return torch.tensor([encode(t, vocab, max_len) for t in texts], dtype=torch.long)


def load_vocab(path):
    with open(path) as f:
        return json.load(f)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

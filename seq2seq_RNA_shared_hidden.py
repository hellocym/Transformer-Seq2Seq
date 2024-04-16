from models.transformer import Seq2SeqTransformer
from datasets.peptide import collate_fn
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
import math
from models.transformer_pytorch import Transformer
import torch.nn as nn
import torch
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import wandb


from utils.utils import *
from datasets.peptide import PeptideDataset


# 创建数据集实例
root = 'data'
files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('csv')]
train_dataset = PeptideDataset(files, split='train', transform=text_transform)
val_dataset = PeptideDataset(files, split='val', transform=text_transform)
test_dataset = PeptideDataset(files, split='test', transform=text_transform)


SRC_VOCAB_SIZE = 256
TGT_VOCAB_SIZE = 256
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
HID_LEN = 10
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 100
LR = 1e-4

wandb.init(
    # set the wandb project where this run will be logged
    project="NMT-RNA",

    # track hyperparameters and run metadata
    config={
        "embedding_size": EMB_SIZE,
        "feed_forward_hidden_size": FFN_HID_DIM,
        "hidden_state_length": HID_LEN,
        "batch_size": BATCH_SIZE,
        "transformer_encoder_layers": NUM_ENCODER_LAYERS,
        "transformer_decoder_layer": NUM_DECODER_LAYERS,
        "learning_rate": LR,
        "architecture": "Transformer",
        "dataset": "RNA",
        "epochs": NUM_EPOCHS,
        "optimizer": 'Adam',
        'cost_function': '0.33*Forward+0.33*Backward+0.33*Hidden'
    }
)

# 示例：使用collate_fn生成dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn)


torch.manual_seed(0)


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)


# helper function to club together sequential operations


def train_epoch(model, optimizer):
    from tqdm import tqdm

    model.train()
    losses = 0

    for src, tgt in tqdm(train_loader, desc="Training"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_input = src[:-1, :]
        # print(src.shape, tgt.shape)

        src_mask1, tgt_mask1, src_padding_mask1, tgt_padding_mask1 = create_mask(
            src, tgt_input)
        src_mask2, tgt_mask2, src_padding_mask2, tgt_padding_mask2 = create_mask(
            tgt, src_input)

        h1, h2, logits1, logits2 = model(src, tgt_input, tgt, src_input,
                                         src_mask1, tgt_mask1, src_mask2, tgt_mask2,
                                         src_padding_mask1, tgt_padding_mask1, src_padding_mask1,
                                         src_padding_mask2, tgt_padding_mask2, src_padding_mask2)

        optimizer.zero_grad()

        tgt_out1 = tgt[1:, :]
        tgt_out2 = src[1:, :]

        loss1 = loss_fn(
            logits1.reshape(-1, logits1.shape[-1]), tgt_out1.reshape(-1))
        loss2 = loss_fn(
            logits2.reshape(-1, logits2.shape[-1]), tgt_out2.reshape(-1))
        loss3 = mse_loss(LT(h1, target_length=HID_LEN),
                         LT(h2, target_length=HID_LEN))
        # loss3 = mse_loss(h1, h2)
        wandb.log({"train_loss_hidden": loss3})
        loss = (loss1 + loss2 + loss3) / 3
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_loader))


def evaluate(model):
    from tqdm import tqdm

    model.eval()
    losses = 0

    for src, tgt in tqdm(val_loader, desc="Validating"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_input = src[:-1, :]

        src_mask1, tgt_mask1, src_padding_mask1, tgt_padding_mask1 = create_mask(
            src, tgt_input)
        src_mask2, tgt_mask2, src_padding_mask2, tgt_padding_mask2 = create_mask(
            tgt, src_input)

        h1, h2, logits1, logits2 = model(src, tgt_input, tgt, src_input,
                                         src_mask1, tgt_mask1, src_mask2, tgt_mask2,
                                         src_padding_mask1, tgt_padding_mask1, src_padding_mask1,
                                         src_padding_mask2, tgt_padding_mask2, src_padding_mask2)

        # print(h1.shape, h2.shape)

        tgt_out1 = tgt[1:, :]
        tgt_out2 = src[1:, :]

        loss1 = loss_fn(
            logits1.reshape(-1, logits1.shape[-1]), tgt_out1.reshape(-1))
        loss2 = loss_fn(
            logits2.reshape(-1, logits2.shape[-1]), tgt_out2.reshape(-1))
        loss = (loss1 + loss2) / 2
        wandb.log({"val_loss_forward": loss1, "val_loss_reverse": loss2})
        losses += loss.item()

    return losses / len(list(val_loader))


best_val_loss = 1e9
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)

    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    wandb.log({'train_loss': train_loss, 'val_loss': val_loss,
              'epoch_time': end_time - start_time})
    if val_loss < best_val_loss:
        torch.save(transformer, os.path.join(
            wandb.run.dir, f"Epoch{epoch}.pth"))
        best_val_loss = val_loss

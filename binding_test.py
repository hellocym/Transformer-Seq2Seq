import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.transformer import Seq2SeqTransformer
from datasets.peptide import ClassificationDataset, collate_fn_classification
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
from pprint import pprint

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


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def evaluate_classification_epoch(model, model_class, loss_fn):
    from tqdm import tqdm

    # model.train()
    model.eval()
    model_class.eval()
    freeze_model(model)
    losses = 0

    for src, tgt, y in tqdm(val_loader, desc="Validating"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        y = y.to(DEVICE)
        # print(y)
        tgt_input = tgt[:-1, :]
        src_input = src[:-1, :]
        # print(src.shape, tgt.shape)

        src_mask1, tgt_mask1, src_padding_mask1, tgt_padding_mask1 = create_mask(
            src, tgt_input)
        src_mask2, tgt_mask2, src_padding_mask2, tgt_padding_mask2 = create_mask(
            tgt, src_input)

        h1, h2, logits1, logits2, logits3, logits4 = model(src, tgt_input, tgt, src_input,
                                                           src_mask1, tgt_mask1, src_mask2, tgt_mask2,
                                                           src_padding_mask1, tgt_padding_mask1, src_padding_mask1,
                                                           src_padding_mask2, tgt_padding_mask2, src_padding_mask2)
        h1_LT, h2_LT = (LT(h1, target_length=HID_LEN).permute(1, 0, 2),
                        LT(h2, target_length=HID_LEN).permute(1, 0, 2))
        # print(h1_LT.shape)

        h1_LT_view = h1_LT.reshape(h1_LT.size(0), -1)
        h2_LT_view = h2_LT.reshape(h2_LT.size(0), -1)
        # print(h1_LT_view.shape)
        input_class_data = torch.cat((h1_LT_view, h2_LT_view), dim=1)
        # print(input_class_data.shape)
        class_out = model_class(input_class_data).squeeze(-1)
        # pprint(y.shape)
        loss = loss_fn(class_out, y)
        # print(loss)

        # wandb.log({"train_loss_hidden": loss3})
        # loss = (loss1 + loss2 + loss3) / 3

        losses += loss.item()

    return losses / len(list(val_loader))


def test_classification(model, model_class):
    from tqdm import tqdm

    model.eval()
    model_class.eval()
    freeze_model(model)
    T = 0
    cnt = 0

    for src, tgt, y in (pbar := tqdm(test_loader, desc="Testing")):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        y = y.to(DEVICE)
        # print(y)
        tgt_input = tgt[:-1, :]
        src_input = src[:-1, :]
        # print(src.shape, tgt.shape)

        src_mask1, tgt_mask1, src_padding_mask1, tgt_padding_mask1 = create_mask(
            src, tgt_input)
        src_mask2, tgt_mask2, src_padding_mask2, tgt_padding_mask2 = create_mask(
            tgt, src_input)

        h1, h2, logits1, logits2, logits3, logits4 = model(src, tgt_input, tgt, src_input,
                                                           src_mask1, tgt_mask1, src_mask2, tgt_mask2,
                                                           src_padding_mask1, tgt_padding_mask1, src_padding_mask1,
                                                           src_padding_mask2, tgt_padding_mask2, src_padding_mask2)
        h1_LT, h2_LT = (LT(h1, target_length=HID_LEN).permute(1, 0, 2),
                        LT(h2, target_length=HID_LEN).permute(1, 0, 2))
        # print(h1_LT.shape)

        h1_LT_view = h1_LT.reshape(h1_LT.size(0), -1)
        h2_LT_view = h2_LT.reshape(h2_LT.size(0), -1)
        # print(h1_LT_view.shape)
        input_class_data = torch.cat((h1_LT_view, h2_LT_view), dim=1)
        # print(input_class_data.shape)
        class_out = model_class(input_class_data).squeeze(-1)
        # pprint(y.shape)
        result = (class_out > 0.5) & (y == 1)
        # print(result)
        T += result.sum().cpu().item()
        # print(T)
        cnt += class_out.shape[0]
        current_average_acc = T / cnt
        pbar.set_description(
            f"Calculating classification Acc - Current Avg: {current_average_acc:.4f}")
        pbar.update(1)

    return T / cnt


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    # 创建数据集实例
    root = '/data/Transformer-Seq2Seq/data_classification/'
    files = [os.path.join(root, f)
             for f in os.listdir(root) if f.endswith('csv')]

    test_dataset = ClassificationDataset(
        files, split='test', transform=text_transform)

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(
        '/data/Transformer-Seq2Seq/wandb/run-20240423_000758-oldrn0nc/files/Epoch8.pth')
    classification_model = BinaryClassifier(2*HID_LEN*EMB_SIZE, HID_LEN, 1)
    classification_model, optimizer, _ = load_checkpoint(
        args.model_path, classification_model, optimizer)

    if torch.cuda.is_available():
        # 将模型移动到 GPU 上
        classification_model.cuda()  # Classification
        model.cuda()  # encoder

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn_classification)
    test_acc = test_classification(model, classification_model)
    print(f'Accuracy on testset:{test_acc}')

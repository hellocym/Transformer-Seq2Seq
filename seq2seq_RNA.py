from timeit import default_timer as timer
from torch.utils.data import DataLoader
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


import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


class PeptideDataset(Dataset):
    def __init__(self, files, split='train', transform=None):
        # 初始为空的列表，用于存储各个分割的数据
        train_data_list = []
        val_data_list = []
        test_data_list = []

        # 对每个文件单独进行划分
        for file in files:
            # 读取文件
            data = pd.read_csv(file, header=None, skiprows=1)
            # 取第一列和第二列数据
            data = data.iloc[:, :2]

            # 划分数据
            train_temp, temp_data = train_test_split(
                data, test_size=0.3, random_state=42)
            val_temp, test_temp = train_test_split(
                temp_data, test_size=1/3, random_state=42)

            # 追加到相应的列表中
            train_data_list.append(train_temp)
            val_data_list.append(val_temp)
            test_data_list.append(test_temp)

        # 合并来自所有文件的数据
        if split == 'train':
            self.data = pd.concat(train_data_list, ignore_index=True)
        elif split == 'val':
            self.data = pd.concat(val_data_list, ignore_index=True)
        elif split == 'test':
            self.data = pd.concat(test_data_list, ignore_index=True)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        column1 = row[0]
        column2 = row[1]
        if self.transform:
            column1 = self.transform(column1)
            column2 = self.transform(column2)
        return (column1, column2)

# 用于字符到整数的映射


def text_transform(text):
    bos_token = 2
    # 首先将文本转换为字符的ascii值列表
    transformed_text = [ord(char) for char in text]
    return transformed_text


# 创建数据集实例
root = 'data'
files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('csv')]
train_dataset = PeptideDataset(files, split='train', transform=text_transform)
val_dataset = PeptideDataset(files, split='val', transform=text_transform)
test_dataset = PeptideDataset(files, split='test', transform=text_transform)


def collate_fn(batch):
    pad_token = 1
    bos_token = 2

    # 处理batch中的每个样本，样本是(column1, column2)的形式
    batch_column1 = [torch.tensor([bos_token] + item[0])
                     for item in batch]  # 对第一列应用转换
    batch_column2 = [torch.tensor([bos_token] + item[1])
                     for item in batch]  # 对第二列应用转换

    # 对两列数据进行padding
    column1_padded = pad_sequence([torch.tensor(x) for x in batch_column1],
                                  padding_value=pad_token, batch_first=True)
    column2_padded = pad_sequence([torch.tensor(x) for x in batch_column2],
                                  padding_value=pad_token, batch_first=True)

    return column1_padded.T, column2_padded.T


# 示例：使用collate_fn生成dataloader
train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32,
                        shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, collate_fn=collate_fn)


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator1 = nn.Linear(emb_size, tgt_vocab_size)
        self.generator2 = nn.Linear(emb_size, src_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src1: Tensor,
                trg1: Tensor,
                src2: Tensor,
                trg2: Tensor,
                src_mask1: Tensor,
                tgt_mask1: Tensor,
                src_mask2: Tensor,
                tgt_mask2: Tensor,
                src_padding_mask1: Tensor,
                tgt_padding_mask1: Tensor,
                memory_key_padding_mask1: Tensor,
                src_padding_mask2: Tensor,
                tgt_padding_mask2: Tensor,
                memory_key_padding_mask2: Tensor):

        src_emb1 = self.positional_encoding(self.src_tok_emb(src1))
        tgt_emb1 = self.positional_encoding(self.tgt_tok_emb(trg1))
        src_emb2 = self.positional_encoding(self.tgt_tok_emb(src2))
        tgt_emb2 = self.positional_encoding(self.src_tok_emb(trg2))
        h1, h2, out1, out2 = self.transformer(src_emb1, tgt_emb1, src_emb2, tgt_emb2,
                                              src_mask1, tgt_mask1, src_mask2, tgt_mask2,
                                              None,
                                              src_padding_mask1, tgt_padding_mask1, memory_key_padding_mask1,
                                              src_padding_mask2, tgt_padding_mask2, memory_key_padding_mask2)

        return h1, h2, self.generator1(out1), self.generator2(out2)

    def encode1(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder1(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode1(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder1(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)

    def encode2(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder2(self.positional_encoding(
            self.tgt_tok_emb(src)), src_mask)

    def decode2(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder2(self.positional_encoding(
            self.src_tok_emb(tgt)), memory,
            tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


torch.manual_seed(0)

SRC_VOCAB_SIZE = 256
TGT_VOCAB_SIZE = 256
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


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
        # loss3 = mse_loss(h1, h2)
        loss = (loss1 + loss2) / 2
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_loader))


def evaluate(model):
    from tqdm import tqdm

    model.eval()
    losses = 0

    for src, tgt in tqdm(val_loader, desc="Training"):
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
        # loss3 = mse_loss(h1, h2)
        loss = (loss1 + loss2) / 2

        losses += loss.item()

    return losses / len(list(val_loader))


NUM_EPOCHS = 100


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save(transformer, f'Epoch{epoch}.pth')

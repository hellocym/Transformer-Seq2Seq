import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.transformer import Seq2SeqTransformer
from utils.utils import generate_square_subsequent_mask
from utils.utils import PositionalEncoding, TokenEmbedding

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(
    './wandb/run-20240413_233133-ypa9goic/files/Epoch47.pth')

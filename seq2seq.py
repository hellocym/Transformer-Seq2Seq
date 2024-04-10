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


# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer(
    'spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer(
    'spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(
        split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


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

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
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

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               # Numericalization
                                               vocab_transform[ln],
                                               tensor_transform)  # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    from tqdm import tqdm

    model.train()
    losses = 0
    train_iter = Multi30k(
        split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm(train_dataloader, desc="Training"):
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

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(
        SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(
        val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
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

    return losses / len(list(val_dataloader))


NUM_EPOCHS = 1


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

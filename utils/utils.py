import torch
from torch import nn
from torch import Tensor
import math
import torch.nn.functional as F


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def text_transform(text):
    bos_token = 2
    # 首先将文本转换为字符的ascii值列表
    transformed_text = [ord(char) for char in text]
    return transformed_text


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


def LT(x, target_length, sigma=1.0):
    # x is expected to have shape (length, batch, dim)
    length, batch, dim = x.shape

    # Creating the squared distance matrix for a range of indices
    j = torch.arange(target_length).unsqueeze(1).repeat(
        1, length)   # Shape: (target_length, length)
    k = torch.arange(length).unsqueeze(0).repeat(
        target_length, 1)  # Shape: (target_length, length)
    # Shape: (target_length, length)
    squared_distance = -((k - j)**2) / (2 * sigma**2)

    # Broadcasting squared_distance across batch dimension
    # Shape: (target_length, length, batch)
    a = squared_distance.unsqueeze(2).repeat(1, 1, batch).to(DEVICE)

    # Softmax across the 'length' dimension
    w = F.softmax(a, dim=1)

    # Re-arranging x to perform batched matrix multiplication: (batch, dim, length)
    x_perm = x.permute(1, 2, 0)

    # Matrix multiplication along the specified dimensions: (batch, dim, target_length)
    z = torch.bmm(x_perm, w.permute(2, 1, 0))

    # Re-arrange back to the desired output shape: (target_length, batch, dim)
    z = z.permute(2, 0, 1)

    return z


# Save checkpoint
def save_checkpoint(model, optimizer, epoch, best_val_loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, filepath)

# Function to load checkpoint


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    scalars = {}
    if isinstance(checkpoint, type({})):
        if 'model_state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'epoch' in checkpoint.keys():
            epoch = checkpoint['epoch']
            scalars['epoch'] = epoch
        if 'best_val_loss' in checkpoint.keys():
            best_val_loss = checkpoint['best_val_loss']
            scalars['best_val_loss'] = best_val_loss
    else:
        model = checkpoint
    return model, optimizer, scalars

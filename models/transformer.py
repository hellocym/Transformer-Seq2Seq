from torch import nn
from torch import Tensor
from models.transformer_pytorch import Transformer
from utils.utils import PositionalEncoding, TokenEmbedding


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
                                       dropout=dropout,
                                       output_attentions=True)
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
        out3 = self.decode2(trg2, self.encode1(src1, src_mask1), tgt_mask2)
        out4 = self.decode1(trg1, self.encode2(src2, src_mask2), tgt_mask1)
        return h1, h2, self.generator1(out1), self.generator2(out2), self.generator2(out3), self.generator1(out4)

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

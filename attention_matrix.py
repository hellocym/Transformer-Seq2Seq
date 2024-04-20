import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.transformer import Seq2SeqTransformer
from utils.utils import *


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, direction=1):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    if direction == 1:
        memory = model.encode1(src, src_mask)
    else:
        memory = model.encode2(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        if direction == 1:
            out = model.decode1(ys, memory, tgt_mask)
        else:
            out = model.decode2(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        if direction == 1:
            prob = model.generator1(out[:, -1])
        else:
            prob = model.generator2(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            # print('hitting end of seq mark.')
            break
    return ys


def infer(model, src_sequence, direction=1):
    model.eval()
    src_sequence = [ord(c) for c in src_sequence]
    src = torch.tensor([BOS_IDX] + src_sequence + [EOS_IDX],
                       dtype=torch.long).to(DEVICE).view(-1, 1)
    # print(src.shape)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=100, start_symbol=BOS_IDX, direction=direction).flatten()

    print(tgt_tokens)

    return ''.join([chr(i) for i in tgt_tokens if i not in [PAD_IDX, BOS_IDX, EOS_IDX]])


def main():
    parser = argparse.ArgumentParser(
        description='Greedy Decoder for Seq2Seq Model')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Input sequence for inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model')
    parser.add_argument('--direction', type=int, default=1,
                        help='Direction of inference: 1 for encode1-decode1, 2 for encode2-decode2')
    args = parser.parse_args()

    # 假设 transformer 已被加载并准备好
    transformer = torch.load(args.model_path)

    output_sequence = infer(transformer, args.sequence,
                            direction=args.direction)
    print("Input Sequence:", args.sequence)
    print("Output Sequence:", output_sequence)


if __name__ == '__main__':
    main()

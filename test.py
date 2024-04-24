import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.transformer import Seq2SeqTransformer
from utils.utils import *
from datasets.peptide import collate_fn, PeptideDataset
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import os
from tqdm import tqdm

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    # print(tgt_tokens)

    return ''.join([chr(i) for i in tgt_tokens if i not in [PAD_IDX, BOS_IDX, EOS_IDX]])


def calculate_bleu(data_loader, model, direction=2):
    model.eval()
    references = []
    hypotheses = []
    batch_bleu_score = 0
    i = 0
    for src, tgt in (pbar := tqdm(data_loader, desc="Testing BLEU")):
        ground_truth = src[0]
        predicted = infer(model, tgt[0], direction)
        # print(ground_truth)
        # print(predicted)

        score = sentence_bleu(
            [list(ground_truth)], list(predicted), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
        batch_bleu_score += score
        i += 1

        current_average_bleu = batch_bleu_score / i
        pbar.set_description(
            f"Calculating BLEU - Current Avg: {current_average_bleu:.4f}")
        pbar.update(1)  # 更新进度条

        # if i > 3:
        #     break

    return batch_bleu_score / i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--sequence", type=str, required=False)
    # parser.add_argument("--direction", type=int, default=2)
    args = parser.parse_args()

    transformer = torch.load(args.model_path)
    # output_sequence = infer(transformer, args.sequence, direction=args.direction)
    # print("Input Sequence:", args.sequence)
    # print("Output Sequence:", output_sequence)
    root = '/data/Transformer-Seq2Seq/data'
    files = [os.path.join(root, f)
             for f in os.listdir(root) if f.endswith('csv')]
    test_dataset = PeptideDataset(
        files, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1)
    bleu_score = calculate_bleu(
        test_loader, transformer)
    print(f"BLEU Score: {bleu_score}")


if __name__ == '__main__':
    main()

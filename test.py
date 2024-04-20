import os
from utils.utils import *
from datasets.peptide import PeptideDataset
from torch.utils.data import DataLoader
from datasets.peptide import collate_fn
import torch
import argparse


BATCH_SIZE = 128

# 创建数据集实例
root = 'data'
files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('csv')]
test_dataset = PeptideDataset(files, split='test', transform=text_transform)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def test(model):
    from tqdm import tqdm

    model.eval()
    losses = 0

    for src, tgt in (t := tqdm(test_loader, desc="Testing")):
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

        tgt_out1 = tgt[1:, :]
        tgt_out2 = src[1:, :]

        loss1 = loss_fn(
            logits1.reshape(-1, logits1.shape[-1]), tgt_out1.reshape(-1))
        loss2 = loss_fn(
            logits2.reshape(-1, logits2.shape[-1]), tgt_out2.reshape(-1))
        loss = (loss1 + loss2) / 2
        t.set_postfix(loss_f=loss1.item(), loss_b=loss2.item())
        # ({"val_loss_forward": loss1, "val_loss_reverse": loss2})
        losses += loss.item()

    print(f'Test Loss: {losses / len(list(test_loader))}')


def main():
    parser = argparse.ArgumentParser(
        description='Greedy Decoder for Seq2Seq Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model')
    args = parser.parse_args()

    # 假设 transformer 已被加载并准备好
    transformer = torch.load(args.model_path)

    test(transformer)


if __name__ == '__main__':
    main()

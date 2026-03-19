import argparse
import pickle
import time
import os

import torchvision
import torch
import torch.multiprocessing as mp

from stepsagnostic import train
from models import AgnosticModel
from torch.multiprocessing import Process
import os

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="exp/    /")

parser.add_argument("--train-mixed", type=str, default="vcf/train_4cate/")
parser.add_argument("--valid-mixed", type=str, default="vcf/test_4cate/")
parser.add_argument("--train-ref-panel", type=str, default="vcf/REF_4cate/")
parser.add_argument("--valid-ref-panel", type=str, default="vcf/REF_4cate/")

parser.add_argument("--query", '-q', default=False)
parser.add_argument("--reference", '-r', default=False)
parser.add_argument("--map", '-m', default=False)

parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--update-every", type=int, default=1)

parser.add_argument("--smoother", type=str, choices=["anc1conv", "none"],
                    default="anc1conv")

parser.add_argument("--ref-pooling", type=str, choices=["maxpool", "topk", "average",
                                                        "baggingmaxpool", "topk-average"],
                    default="topk-average")

parser.add_argument("--topk-k", type=int, default=1)

parser.add_argument("--win-size", type=int, default=4096) #8192
parser.add_argument("--win-stride", type=int, default=-1)

parser.add_argument("--dropout", type=float, default=-1)

parser.add_argument("--loss", type=str, default="FC", choices=["CE"])

parser.add_argument("--resume", dest="resume", action='store_true', default=True)

parser.add_argument("--validate", default=False)

parser.add_argument("--n-refs", type=int, default=99999)

parser.add_argument("--comment", type=str, default=None)

parser.add_argument("--initial", default=True)


def run(fn, model, args, world_size):
    mp.spawn(fn,
             args=(model, args, world_size),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    args = parser.parse_args()
    lr = args.lr
    validd = args.validate
    loss = args.loss
    batch_size = args.batch_size
    exp = args.exp

    if args.resume:
        assert (bool(args.exp))
        with open(f"{args.exp}/args.pckl", "rb") as f:
            args = pickle.load(f)
            args.lr = lr
            args.validate = validd
            args.loss = loss
            args.batch_size = batch_size
            args.resume = True
            args.exp = exp
            args.initial = False

    print(args)

    model = AgnosticModel(args)

    if args.initial:
        print("[INFO] Initializing model from scratch (no checkpoint loaded).")

    if not args.resume:
        if not os.path.isdir(args.exp):
            os.mkdir(args.exp)
            os.mkdir(args.exp + "/models")

    with open(args.exp + "/args.pckl", "wb") as f:
        pickle.dump(args, f)

    world_size = 1
    train(0, model, args, world_size)


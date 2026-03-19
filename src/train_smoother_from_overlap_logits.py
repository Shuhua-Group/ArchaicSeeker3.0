import argparse, pickle, os, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from stepsagnostic.utils import (
    ProgressSaver, ReshapedCrossEntropyLoss
)
from stepsagnostic import train_smoother
from models import AncestryLevelConvSmoother

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="exp/Smoother_Only_MambaBased_4096_MultiStride_Kernel_8192_V6_20250509_lr1e5_V2/")
parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b","--batch-size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--dropout", type=float, default=0.15)
parser.add_argument("--loss", choices=["FC","CE"], default="FC")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--validate",default=False)
parser.add_argument("--inference_pt",
    default="/cpfs01/projects-HDD/humPOG_HDD/public/leichang/ArchaicSim/00_BaseModel_Pretrain_InferData/00_Mamba_Win4096_allData_Train_20250428_Latest_different_stride/")
parser.add_argument("--inference_val_pt",
    default="/cpfs01/projects-HDD/humPOG_HDD/public/leichang/ArchaicSim/00_BaseModel_Pretrain_InferData/00_Mamba_Win4096_allData_Val_20250428_latest/")
parser.add_argument("--inference_val_small_pt",
    default="/cpfs01/projects-HDD/humPOG_HDD/public/leichang/ArchaicSim/00_BaseModel_Pretrain_InferData/00_Mamba_Win4096_allData_Val_20250428_latest/valid_loaders_small/")
parser.add_argument("--train-strides", type=int, nargs="+", default=[256,512,1024,2048])
parser.add_argument("--val-strides", type=int, nargs="+", default=[256,512,1024,2048])
parser.add_argument("--train-ratio", type=float, default=0.75)
parser.add_argument("--val-ratio", type=float, default=0.25)
parser.add_argument("--n-refs", type=int, default=99999)
parser.add_argument("--comment", type=str, default=None)
parser.add_argument("--update-every", type=int, default=1)
parser.add_argument("--win-size", type=int, default=4096)
parser.add_argument("--lr-decay", type=int, default=-1)
parser.add_argument("--smoother", type=str, choices=["anc1conv", "none"],
                    default="anc1conv")

parser.add_argument("--ref-pooling", type=str, choices=["maxpool", "topk", "average",
                                                        "baggingmaxpool","topk-average"],
                    default="topk-average")
args = parser.parse_args()

class SmootherDatasetNoMerge(torch.utils.data.Dataset):
    def __init__(self, pt):
        obj = torch.load(pt, map_location="cpu")
        self.pred_list, self.pos_list = obj["predictions"], obj["positions"]
        self.lbl_list,  self.arc_list = obj["labels"], obj["single_arc"]
    def __len__(self): return len(self.pred_list)
    def __getitem__(self, i):
        return (self.pred_list[i], self.pos_list[i],
                self.lbl_list[i],  self.arc_list[i])

def collate(batch):
    preds, poss, lbls, arcs = zip(*batch)
    preds = torch.cat(preds, 0)
    pos   = torch.cat(poss, 0)  if None not in poss else None
    lbl   = torch.cat(lbls, 0)  if None not in lbls else None
    arc   = torch.stack([a.squeeze() if torch.is_tensor(a) else torch.tensor(a)
                         for a in arcs])
    return preds, pos, lbl, arc

def list_pt_files(root: Path):
    res=[]
    for p in root.rglob("*.pt"):
        if p.name.startswith("s") and "_" in p.name:
            st=int(p.name.split("_")[0][1:])
        elif p.parent.name.startswith("stride"):
            st=int(p.parent.name.replace("stride",""))
        else:
            st=-1
        res.append((st, p))
    return res

def build_loaders(root_dir, stride_list, ratio, batch):
    root = Path(root_dir)
    files = list_pt_files(root)
    by_stride={}
    for st, p in files:
        if st in stride_list:
            by_stride.setdefault(st, []).append(p)
    loaders=[]
    for st, flist in by_stride.items():
        keep = flist if ratio>=1.0 else random.sample(flist, max(1,int(len(flist)*ratio)))
        for fp in keep:
            dl = DataLoader(SmootherDatasetNoMerge(fp), batch_size=batch,
                            shuffle=True, pin_memory=True, collate_fn=collate)
            loaders.append(dl)
    random.shuffle(loaders)
    return loaders

def main():
    print("Args:", args)
    model = AncestryLevelConvSmoother(in_planes=4, planes=32,
                                      kernel_size=8192, dropout_p=args.dropout)
    criterion = ReshapedCrossEntropyLoss(args.loss)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ld = build_loaders(args.inference_pt, args.train_strides,
                             args.train_ratio, args.batch_size)
    val_ld   = build_loaders(args.inference_val_pt, args.val_strides,
                             args.val_ratio,   args.batch_size)
    val_small_ld = build_loaders(args.inference_val_small_pt,
                                 args.val_strides, args.val_ratio,
                                 args.batch_size)

    (Path(args.exp)/"models").mkdir(parents=True, exist_ok=True)
    if not args.resume:
        with open(Path(args.exp)/"args.pckl","wb") as f: pickle.dump(args,f)
    saver = ProgressSaver(args.exp)

    train_smoother(args=args, model=model,
                   train_loaders=train_ld,
                   valid_loaders=val_ld,
                   valid_loaders_small=val_small_ld,
                   criterion=criterion, optimizer=optim,
                   progress_saver=saver, rank=0)

if __name__ == "__main__":
    main()


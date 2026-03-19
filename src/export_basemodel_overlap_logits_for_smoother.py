import argparse
import pickle
import time
import os
import shutil
import warnings

import torchvision
import torch
import torch.multiprocessing as mp

from stepsagnostic import (
    train,
    baseInferDataLoad,
    inference_and_save_basemodel_overlap,
    baseInferDataLoadValVersion,
)
from models import AgnosticModel, AgnosticModelInferUsed, AgnosticModelInferNoSmoother
from torch.multiprocessing import Process

parser = argparse.ArgumentParser(description="ArchaicSeeker3.0")

parser.add_argument(
    "--model-cp",
    type=str,
    default="/cpfs01/projects-HDD/humPOG_HDD/public/leichang/ArchaicSim/exp/"
            "basemodel_pretrain_winsize500_mamba_20250417/models/best_model.pth",
    help="Path to the model checkpoint file"
)
parser.add_argument(
    "--model-args",
    type=str,
    default=None,
    help="Additional arguments for the model, specified in pckl format"
)
parser.add_argument(
    "--stride",
    type=int,
    nargs="+",
    default=[256, 512, 1024, 2048, 4096, 8192],
    help="List of strides to run inference with (default 512)"
)
parser.add_argument(
    "--out-folder", "-o",
    default="/cpfs01/projects-HDD/humPOG_HDD/public/leichang/ArchaicSim/"
            "00_BaseModel_Pretrain_InferData/"
            "00_Mamba_Win4096_allData_Val_20250428_latest/",
    help="Output folder for the results"
)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    if not args.model_args:
        if "best_model" in args.model_cp:
            args.model_args = args.model_cp.replace('models/best_model.pth', 'args.pckl')
        else:
            args.model_args = args.model_cp.replace('models/last_model.pth', 'args.pckl')
    for stride in args.stride:
        print(f"\n===== RUNNING VAL INFERENCE WITH stride = {stride} =====")

        with open(args.model_args, "rb") as f:
            model_args = pickle.load(f)
        model_args.win_stride = stride
        print(f"[INFO] win_stride = {model_args.win_stride}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AgnosticModelInferNoSmoother(model_args)
        checkpoint = torch.load(args.model_cp, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

        valid_loaders, valid_loaders_small, infos = baseInferDataLoadValVersion(model_args)

        tmp_dir_small = os.path.join(args.out_folder, f".tmp_val_small_s{stride}")
        os.makedirs(tmp_dir_small, exist_ok=True)
        inference_and_save_basemodel_overlap(
            model=model,
            data_loaders=valid_loaders_small,
            infos=infos,
            args=args,
            output_dir=tmp_dir_small,
        )

        tmp_dir = os.path.join(args.out_folder, f".tmp_val_s{stride}")
        os.makedirs(tmp_dir, exist_ok=True)
        inference_and_save_basemodel_overlap(
            model=model,
            data_loaders=valid_loaders,
            infos=infos,
            args=args,
            output_dir=tmp_dir,
        )

        small_out = os.path.join(args.out_folder, "valid_loaders_small")
        os.makedirs(small_out, exist_ok=True)
        for fname in os.listdir(tmp_dir_small):
            if not fname.endswith(".pt"):
                continue
            src = os.path.join(tmp_dir_small, fname)
            dst = os.path.join(small_out, f"s{stride}_{fname}")
            shutil.move(src, dst)
            print(f"[MOVED] {src} -> {dst}")

        for fname in os.listdir(tmp_dir):
            if not fname.endswith(".pt"):
                continue
            src = os.path.join(tmp_dir, fname)
            dst = os.path.join(args.out_folder, f"s{stride}_{fname}")
            shutil.move(src, dst)
            print(f"[MOVED] {src} -> {dst}")

        shutil.rmtree(tmp_dir_small)
        shutil.rmtree(tmp_dir)
        print(f"[DONE] stride {stride}: all val files moved into {args.out_folder}")

    print("\nALL VAL INFERENCE DONE.")


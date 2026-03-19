import torch
import torch.nn.functional as f

from .utils import ancestry_accuracy, ProgressSaver, AverageMeter, ReshapedCrossEntropyLoss, \
    adjust_learning_rate, to_device, correct_max_indices, compute_ibd, ancestry_metrics,build_transforms,ancestry_metrics_ad,\
        find_introgression_segments, ancestry_metrics_label_based
from ..dataloaders import ReferencePanelDataset, reference_panel_collate, ReferencePanelDatasetSmall
from torch.nn.parallel import  DistributedDataParallel as ddp
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Sampler
import time
import os
from tqdm import tqdm
import random   
import numpy as np
import pandas as pd

loss_proportion = 0.9

def train_smoother(
    args,
    model, 
    train_loaders, 
    valid_loaders, 
    valid_loaders_small,
    criterion, 
    optimizer,
    progress_saver,
    rank=0
):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loss_meter = AverageMeter()
    best_val_loss = float('inf')
    best_val_f1   = 0.0
    best_epoch    = 0
    init_epoch    = 0
    iters         = 0
    init_time     = time.time()

    if args.resume:
        progress_saver.load_progress()
        init_epoch, best_val_loss, iters, start_time = progress_saver.get_resume_stats()
        init_time = time.time() - start_time

        model.load_state_dict(torch.load(args.exp + "/models/best_model.pth"))

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"loaded state dict from epoch {init_epoch}")
        init_epoch += 1

    if args.validate:
        val_acc, val_recall, val_pre, val_f1, val_loss = smoother_validate(model, valid_loaders, criterion, init_epoch-1, args)
        epoch_data = {
            "epoch": init_epoch-1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_recall": val_recall,
            "val_precision": val_pre,
            "val_f1": val_f1,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            # "lr": ...
        }
        print(epoch_data)
        return

    for n in range(init_epoch, args.num_epochs):
        model.train()
        train_loss_meter.reset()

        if args.lr_decay > 0:
            lr = adjust_learning_rate(args.lr, args.lr_decay, optimizer, n)
        else:
            lr = args.lr

        start_t = time.time()
        for loader_idx in tqdm(range(len(train_loaders)), 
                      desc="🌐 Loaders", 
                      position=0,  
                      leave=True):
            train_loader = train_loaders[loader_idx]

            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, batch in pbar:
                iters += 1
                # warmup or dynamic LR
                update_lr(optimizer, iters, 400, 1e-5, args.lr)
                preds, pos, lbl, single_arc = batch
                preds = preds.to(device).float()

                if pos is not None:
                    pos = pos.to(device).float()
                if lbl is not None:
                    lbl = lbl.to(device)

                pos = (pos / 1000000 / 100).to(preds.device) + torch.zeros([preds.shape[0],1,preds.shape[2]], dtype=torch.float32).to(preds.device)
                out = model.forward(preds, pos=pos)  # Smoother forward
                if lbl is not None:
                    lbl[lbl==3] = 2
                    if single_arc is not None: 
                        if single_arc.item() == 1:
                            lbl[lbl == 2] = 1
                    lbl = lbl.unsqueeze(dim=1)

                if lbl is not None:
                    # shape match => e.g. out[:,0,:] => shape(B,L)
                    loss = criterion(out, lbl, n) 
                else:
                    loss = torch.tensor(0.0, device=device)

                loss.backward()
                if ((i + 1) % args.update_every) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss_meter.update(loss.item())
                pbar.set_description(f"Training (loss: {train_loss_meter.get_average():.5f}, lr: {optimizer.param_groups[0]['lr']})")
                
        if rank==0:
            val_acc, val_recall, val_pre, val_f1, val_loss = smoother_validate(model, valid_loaders, criterion, n, args)
            train_loss_val = train_loss_meter.get_average()
            total_time = time.time() - init_time

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                best_epoch = n
                torch.save(model.state_dict(), args.exp + "/models/best_model.pth")

            torch.save(model.state_dict(), args.exp + "/models/last_model.pth")
            torch.save(optimizer.state_dict(), args.exp + "/models/last_optim.pth")

            epoch_data = {
                "epoch": n,
                "train_loss": train_loss_val,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_recall": val_recall,
                "val_precision": val_pre,
                "val_f1": val_f1,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "time": total_time,
                "lr": lr,
                "iter": iters
            }
            progress_saver.update_epoch_progess(epoch_data)
            print(epoch_data)

    print(f"Finished training. Best epoch={best_epoch}, best_val_loss={best_val_loss}, best_val_f1={best_val_f1}")


def update_lr(optimizer, total_iters, warmup_iters, warmup_start_lr, initial_lr):
    if total_iters < warmup_iters:
        lr = warmup_start_lr + (initial_lr - warmup_start_lr) * (total_iters / warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = initial_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(rank, model, args, world_size):
    # setup(rank, world_size)

    transforms = build_transforms(args)

    print("Loading train data")
    paths = ['DenSplitTime_301000_DenAdmixTime_50000_DenProp_0.02']
    ratios = [0.1, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.6, 0.77, 1.0]
    ratios2 = [0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.0]
    train_datasets = []
    
    paths2 = ["mut_1.2e-08_rec_1.2e-08"]
    
    paths3 = ['afr10_2anc']
    
    paths4 = ['afr100_GeneticMap_2']
    
    for path in paths[:36]: # paths[:60]: # [paths[i] for i in [0,6,13,19,28,33,38,43,48,53]] [paths[i] for i in [0, 5, 9, 12, 17, 21, 24, 30, 36, 42, 48, 54, 60, 61]]
        single_arc = 0
        if "DenSplitTime" in path:
            single_arc = 2
        elif "NeanSplitTime" in path:
            single_arc = 1
        for i in [1,2,3,4,5,10]: #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc)
            train_datasets.append(train_dataset)
            # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            # train_loaders.append(train_loader)

    for path in paths2:
        single_arc = 0
        for i in range(10, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc)
            train_datasets.append(train_dataset)
            # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            # train_loaders.append(train_loader)

    for path in paths3:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=32)
            train_datasets.append(train_dataset)
            # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            # train_loaders.append(train_loader)
    for path in paths4:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                genetic_map="vcf/afr100_GeneticMap/geneticMap/plink.chr22.GRCh37.map",
                                                n_samples=32)
            train_datasets.append(train_dataset)

    concatenated_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(concatenated_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=reference_panel_collate)
    # batch_sampler = CustomBatchSampler(train_datasets, batch_size=args.batch_size)
    # train_loader = DataLoader(concatenated_dataset, batch_sampler=batch_sampler, collate_fn=reference_panel_collate)


    train_loaders = [train_loader]
    print("Loading validation data")
    valid_loaders = []
    valid_loaders_small = []
    infos = []
    for path in paths4:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                genetic_map="vcf/afr100_GeneticMap/geneticMap/plink.chr22.GRCh37.map",
                                                n_samples=32)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)

    for path in paths3:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.valid_mixed + path + "_test%d_vcf_and_labels_test_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.valid_ref_panel + path + "_test%d_vcf_and_labels_REF_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=10)

            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)

    for path in paths2:
        single_arc = 0
        for i in range(10, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.valid_mixed + path + "_test%d_vcf_and_labels_test_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.valid_ref_panel + path + "_test%d_vcf_and_labels_REF_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=1)

            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)

    for path in paths[:36]: #[paths[i] for i in [0,6,13,19,28,33,38,43,48,53]]: # paths[:60]: #  [paths[i] for i in [0, 5, 9, 12, 17, 21, 24, 30, 36, 42, 48, 54, 60, 61]]
        single_arc = 0
        if "DenSplitTime" in path:
            single_arc = 2
        elif "NeanSplitTime" in path:
            single_arc = 1
        for i in [1,2,3,4,5,10]:
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.valid_mixed + path + "_test%d_vcf_and_labels_test_random_"%i+str(ratios[i-1])+".h5",
                                                reference_panel_h5=args.valid_ref_panel + path + "_test%d_vcf_and_labels_REF_random_"%i+str(ratios[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=1)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)
            if path in paths[25:27]:
                valid_loaders_small.append(valid_loader)
    
    print("Data loaded")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(rank)
    print("device:", device)
    # print(torch.cuda.device_count())
    model.to(rank)
    # model = ddp(model, device_ids=[rank],find_unused_parameters=False)
    criterion = ReshapedCrossEntropyLoss(args.loss)

    # Basic
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    init_time = time.time()

    progress_saver = ProgressSaver(args.exp)
    train_loss_meter = AverageMeter()
    best_val_f1 = -1
    best_epoch = -1
    best_val_loss = 999
    iters = 0

    lr = args.lr

    init_epoch = 0
    if args.resume:
        progress_saver.load_progress()
        if 'iter' not in progress_saver.progress:
            progress_saver.progress['iter'] = [4000]
        init_epoch, best_val_loss, iters, start_time = progress_saver.get_resume_stats()
        print(best_val_loss)
        # best_val_loss=  0.1 #0.0030613772839391154

        init_time = time.time() - start_time
        model.load_state_dict(torch.load(args.exp + "/models/best_model.pth"))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(rank)
        print("loaded state dict from epoch %d" % init_epoch)

        init_epoch += 1

    if args.validate:
        val_acc, val_recall, val_pre, val_f1, val_loss = validate(model, valid_loaders, criterion, init_epoch-1, args)
        epoch_data = {
            "epoch": init_epoch-1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_recall": val_recall,
            "val_precision": val_pre,
            "val_f1": val_f1,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "lr": lr
        }

        print(epoch_data)

    else:
        for n in range(init_epoch, args.num_epochs):
            model.train()
            train_loss_meter.reset()
            if args.lr_decay > 0:
                lr = adjust_learning_rate(args.lr, args.lr_decay, optimizer, n)
            start_t = time.time()
            for _ in range(len(train_loaders)):
                train_loader = train_loaders[_]

                pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
                for i, batch in pbar:
                    iters += 1

                    update_lr(optimizer, iters, 400, 1e-5, args.lr)
                    batch = to_device(batch, rank)
                    output = model.forward(batch)
                    label = output['labels']
                    label[label==3]=2
                    arc = batch["single_arc"][0]
                    if arc==1:
                        label[label==2]=1
                    label = label.unsqueeze(dim=1)

                
                    loss = criterion(output["out_basemodel"], label.to(rank),n)
                    loss.backward()
                    if ((i + 1) % args.update_every) == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    train_loss_meter.update(loss.item())
                    pbar.set_description(f"Training (loss: {train_loss_meter.get_average():.5f}, lr: {optimizer.param_groups[0]['lr']})")

                    if rank==0 and iters==400:
                        val_acc, val_recall, val_pre, val_f1, val_loss = validate(model, valid_loaders_small, criterion, n, args)
                        train_loss = train_loss_meter.get_average()

                        total_time = time.time() - init_time
                        if best_val_loss > val_loss:
                            best_val_loss = val_loss
                            best_val_f1 = val_f1
                            best_epoch = n
                            torch.save(model.state_dict(), args.exp + "/models/best_model.pth")

                        torch.save(model.state_dict(), args.exp + "/models/last_model.pth")
                        torch.save(optimizer.state_dict(), args.exp + "/models/last_optim.pth")

                        epoch_data = {
                            "epoch": n,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_recall": val_recall,
                            "val_precision": val_pre,
                            "val_f1": val_f1,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "best_val_f1": best_val_f1,
                            "time": total_time,
                            "lr": lr,
                            "iter": iters
                        }

                        print(epoch_data)


            if rank==0:
                val_acc, val_recall, val_pre, val_f1, val_loss = validate(model, valid_loaders, criterion, n, args)
                train_loss = train_loss_meter.get_average()

                total_time = time.time() - init_time
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_val_f1 = val_f1
                    best_epoch = n
                    torch.save(model.state_dict(), args.exp + "/models/best_model.pth")

                torch.save(model.state_dict(), args.exp + "/models/last_model.pth")
                torch.save(optimizer.state_dict(), args.exp + "/models/last_optim.pth")

                epoch_data = {
                    "epoch": n,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_recall": val_recall,
                    "val_precision": val_pre,
                    "val_f1": val_f1,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_f1": best_val_f1,
                    "time": total_time,
                    "lr": lr,
                    "iter": iters
                }

                progress_saver.update_epoch_progess(epoch_data)

                print(epoch_data)


def inference_and_save_basemodel_overlap(
    model, 
    data_loaders, 
    infos, 
    args, 
    output_path="basemodel_inference.pt"
):
    device = next(model.parameters()).device
    model.eval()

    all_predictions = []
    all_positions   = []
    all_labels      = []
    all_infos       = []

    with torch.no_grad():
        for loader_idx, loader in enumerate(tqdm(data_loaders, desc="AllLoaders", ncols=100)):
            info_this = infos[loader_idx]

            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Loader{loader_idx}", ncols=100, leave=False)):
                out_dict = model(batch, test=True, infer=True)

                preds  = out_dict["predictions"].cpu()
                pad_num= out_dict["pad_num"]
                lbl    = out_dict["labels"]
                if lbl is not None:
                    lbl = lbl.cpu()

                pos_cpu = batch["pos"].cpu() if torch.is_tensor(batch["pos"]) else None

                all_predictions.append(preds)
                all_positions.append(pos_cpu)
                all_labels.append(lbl)
                all_infos.append(info_this)

    save_data = {
        "predictions": all_predictions,
        "positions":   all_positions,
        "labels":      all_labels
    }
    torch.save(save_data, output_path)
    print(f"Inference done. Saved to {output_path}")


def inference_and_save_basemodel_overlap(
    model, 
    data_loaders, 
    infos, 
    args, 
    output_dir="basemodel_inference_results"
):

    import os
    from tqdm import tqdm
    import torch

    device = next(model.parameters()).device
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for loader_idx, loader in enumerate(tqdm(data_loaders, desc="AllLoaders", ncols=100)):
            info_this = infos[loader_idx]

            loader_predictions = []
            loader_positions   = []
            loader_labels      = []

            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Loader{loader_idx}", ncols=100, leave=False)):
                out_dict = model(batch, test=True, infer=True)

                preds  = out_dict["predictions"].cpu()  # (B, C, L) after overlap merge
                pad_num= out_dict["pad_num"]
                lbl    = out_dict["labels"]
                if lbl is not None:
                    lbl = lbl.cpu()

                pos_cpu = None
                if "pos" in batch and torch.is_tensor(batch["pos"]):
                    pos_cpu = batch["pos"].cpu()

                loader_predictions.append(preds)
                loader_positions.append(pos_cpu)
                loader_labels.append(lbl)

            save_data = {
                "predictions": loader_predictions,
                "positions":   loader_positions,
                "labels":      loader_labels
            }

            file_path = os.path.join(output_dir, f"basemodel_inference_loader{loader_idx}.pt")
            torch.save(save_data, file_path)
            print(f"[Loader {loader_idx}] Done. Saved to {file_path}")


def inference_and_save_basemodel_overlap(
    model, 
    data_loaders, 
    infos, 
    args, 
    output_dir="basemodel_inference_results"
):
    import os
    import torch
    from tqdm import tqdm
    print("[INFO]: save with Single_arc par.")
    device = next(model.parameters()).device
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for loader_idx, loader in enumerate(tqdm(data_loaders, desc="AllLoaders", ncols=100)):
            info_this = infos[loader_idx]

            loader_predictions = []
            loader_positions   = []
            loader_labels      = []
            loader_single_arc  = []

            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Loader{loader_idx}", ncols=100, leave=False)):
                out_dict = model(batch, test=True, infer=True)
                preds = out_dict["predictions"].cpu()   # (B,C,L_final) no extra pad
                lbl   = out_dict["labels"]
                single_arc = batch.get("single_arc", None)
                if single_arc is not None:
                    single_arc = single_arc.cpu()  # 保存为 CPU 张量
                else:
                    print("[WARNING]: single_arc is None!")
                
                if lbl is not None:
                    lbl = lbl.cpu()                     # (B,L) same shape as aggregator?

                # pos
                pos_cpu = None
                if "pos" in batch and torch.is_tensor(batch["pos"]):
                    pos_cpu = batch["pos"].cpu()         # (B,L) => should match aggregator?

                loader_predictions.append(preds)
                loader_positions.append(pos_cpu)
                loader_labels.append(lbl)
                loader_single_arc.append(single_arc)

            save_data = {
                "predictions": loader_predictions,
                "positions":   loader_positions,
                "labels":      loader_labels,
                "single_arc":  loader_single_arc
            }
            file_path = os.path.join(output_dir, f"basemodel_inference_loader{loader_idx}.pt")
            torch.save(save_data, file_path)
            print(f"[Loader {loader_idx}] => {file_path}")

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.info = [d.info for d in datasets]  

def baseInferDataLoad(args):
    transforms = build_transforms(args)

    print("Loading train data")
    paths = ['DenSplitTime_301000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_318000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_34000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_42000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.004',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.008',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.012',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.016',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_58000_DenProp_0.02',
            'DenSplitTime_352000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_369000_DenAdmixTime_50000_DenProp_0.02',
            'NeanSplitTime_100000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_105000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_85000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_90000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_34000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_42000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.004',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.008',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.012',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.016',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_58000_NeanProp_0.02',
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.004",
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.008",
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.012",
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.016",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.004",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.008",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.012",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.016",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.004",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.008",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.012",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.016",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.004",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.008",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.012",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.016",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.004",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.008",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.012",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.016",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.004",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.008",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.012",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.016",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.004",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.008",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.012",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.016",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.004",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.008",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.012",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.016",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.004",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.008",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.012",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.016"]
    ratios = [0.1, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.6, 0.77, 1.0]
    ratios2 = [0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.0]
    train_datasets = []
    
    paths2 = ["mut_1.2e-08_rec_1.2e-08",
                "mut_1.2e-08_rec_1.4e-08",
                "mut_1.2e-08_rec_1.6e-08",
                "mut_1.2e-08_rec_1e-8",
                "mut_1.2e-08_rec_8e-09",
                "mut_1.4e-08_rec_1.2e-08",
                "mut_1.4e-08_rec_1.4e-08",
                "mut_1.4e-08_rec_1.6e-08",
                "mut_1.4e-08_rec_1e-8",
                "mut_1.4e-08_rec_8e-09",
                "mut_1.6e-08_rec_1.2e-08",
                "mut_1.6e-08_rec_1.4e-08",
                "mut_1.6e-08_rec_1.6e-08",
                "mut_1.6e-08_rec_1e-8",
                "mut_1.6e-08_rec_8e-09",
                "mut_1e-08_rec_1.2e-08",
                "mut_1e-08_rec_1.4e-08",
                "mut_1e-08_rec_1.6e-08",
                "mut_1e-08_rec_1e-8",
                "mut_1e-08_rec_8e-09",
                "mut_8e-09_rec_1.2e-08",
                "mut_8e-09_rec_1.4e-08",
                "mut_8e-09_rec_1.6e-08",
                "mut_8e-09_rec_1e-8",
                "mut_8e-09_rec_8e-09"]
    
    paths3 = ['afr10_2anc',
            'afr20_2anc',
            'afr30_2anc',
            "HumanNeanderthalDenisovan100",
            "HumanNeanderthalDenisovan50",
            "HumanNeanderthalDenisovan10",
            "Bonobo_100",
            "Bonobo_50",
            "Bonobo_10",
            "BonoboGhost",
            "HumanNeanderthal"]
    
    paths4 = ['afr100_GeneticMap_2',
            'afr100_GeneticMap']

    infos=[]
    train_loaders=[]
    
    for path in paths[:36]: # paths[:60]: # [paths[i] for i in [0,6,13,19,28,33,38,43,48,53]] [paths[i] for i in [0, 5, 9, 12, 17, 21, 24, 30, 36, 42, 48, 54, 60, 61]]
        single_arc = 0
        if "DenSplitTime" in path:
            single_arc = 2
        elif "NeanSplitTime" in path:
            single_arc = 1
        for i in [1,2,3,4,5,10]: #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)

    for path in paths2:
        single_arc = 0
        for i in range(10, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)

    for path in paths3:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=32)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)
            
    for path in paths4:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                genetic_map="vcf/afr100_GeneticMap/geneticMap/plink.chr22.GRCh37.map",
                                                n_samples=32)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)

    print("Data loaded")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return train_loaders, infos


def baseInferDataLoadSmallTest(args):
    transforms = build_transforms(args)

    print("Loading train data")
    paths = ['DenSplitTime_301000_DenAdmixTime_50000_DenProp_0.02']
    ratios = [0.1, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.6, 0.77, 1.0]
    ratios2 = [0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.0]
    train_datasets = []
    
    paths2 = ["mut_1.2e-08_rec_1.2e-08"]
    
    paths3 = ['afr10_2anc']
    
    paths4 = ['afr100_GeneticMap_2']

    infos=[]
    train_loaders=[]
    
    for path in paths[:36]: # paths[:60]: # [paths[i] for i in [0,6,13,19,28,33,38,43,48,53]] [paths[i] for i in [0, 5, 9, 12, 17, 21, 24, 30, 36, 42, 48, 54, 60, 61]]
        single_arc = 0
        if "DenSplitTime" in path:
            single_arc = 2
        elif "NeanSplitTime" in path:
            single_arc = 1
        for i in [1,2,3,4,5,10]: #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)

    for path in paths2:
        single_arc = 0
        for i in range(10, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)

    for path in paths3:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=32)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)
            
    for path in paths4:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            train_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.train_mixed + path + "_test%d_vcf_and_labels_train_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.train_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                genetic_map="vcf/afr100_GeneticMap/geneticMap/plink.chr22.GRCh37.map",
                                                n_samples=32)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            train_loaders.append(train_loader)
            infos.append(train_dataset.info)

    print("Data loaded")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return train_loaders, infos


def baseInferDataLoadValVersion(args):
    transforms = build_transforms(args)

    print("Loading train data")
    paths = ['DenSplitTime_301000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_318000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_34000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_42000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.004',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.008',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.012',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.016',
            'DenSplitTime_335000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_335000_DenAdmixTime_58000_DenProp_0.02',
            'DenSplitTime_352000_DenAdmixTime_50000_DenProp_0.02',
            'DenSplitTime_369000_DenAdmixTime_50000_DenProp_0.02',
            'NeanSplitTime_100000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_105000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_85000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_90000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_34000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_42000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.004',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.008',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.012',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.016',
            'NeanSplitTime_95000_NeanAdmixTime_50000_NeanProp_0.02',
            'NeanSplitTime_95000_NeanAdmixTime_58000_NeanProp_0.02',
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.004",
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.008",
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.012",
            "DenAdmixTime_34000_NeanAdmixTime_34000_DenProp_0.016",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.004",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.008",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.012",
            "DenAdmixTime_34000_NeanAdmixTime_50000_DenProp_0.016",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.004",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.008",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.012",
            "DenAdmixTime_34000_NeanAdmixTime_58000_DenProp_0.016",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.004",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.008",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.012",
            "DenAdmixTime_50000_NeanAdmixTime_34000_DenProp_0.016",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.004",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.008",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.012",
            "DenAdmixTime_50000_NeanAdmixTime_50000_DenProp_0.016",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.004",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.008",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.012",
            "DenAdmixTime_50000_NeanAdmixTime_58000_DenProp_0.016",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.004",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.008",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.012",
            "DenAdmixTime_58000_NeanAdmixTime_34000_DenProp_0.016",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.004",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.008",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.012",
            "DenAdmixTime_58000_NeanAdmixTime_50000_DenProp_0.016",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.004",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.008",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.012",
            "DenAdmixTime_58000_NeanAdmixTime_58000_DenProp_0.016"]
    ratios = [0.1, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.6, 0.77, 1.0]
    ratios2 = [0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.0]
    train_datasets = []
    
    paths2 = ["mut_1.2e-08_rec_1.2e-08",
                "mut_1.2e-08_rec_1.4e-08",
                "mut_1.2e-08_rec_1.6e-08",
                "mut_1.2e-08_rec_1e-8",
                "mut_1.2e-08_rec_8e-09",
                "mut_1.4e-08_rec_1.2e-08",
                "mut_1.4e-08_rec_1.4e-08",
                "mut_1.4e-08_rec_1.6e-08",
                "mut_1.4e-08_rec_1e-8",
                "mut_1.4e-08_rec_8e-09",
                "mut_1.6e-08_rec_1.2e-08",
                "mut_1.6e-08_rec_1.4e-08",
                "mut_1.6e-08_rec_1.6e-08",
                "mut_1.6e-08_rec_1e-8",
                "mut_1.6e-08_rec_8e-09",
                "mut_1e-08_rec_1.2e-08",
                "mut_1e-08_rec_1.4e-08",
                "mut_1e-08_rec_1.6e-08",
                "mut_1e-08_rec_1e-8",
                "mut_1e-08_rec_8e-09",
                "mut_8e-09_rec_1.2e-08",
                "mut_8e-09_rec_1.4e-08",
                "mut_8e-09_rec_1.6e-08",
                "mut_8e-09_rec_1e-8",
                "mut_8e-09_rec_8e-09"]
    
    paths3 = ['afr10_2anc',
            'afr20_2anc',
            'afr30_2anc',
            "HumanNeanderthalDenisovan100",
            "HumanNeanderthalDenisovan50",
            "HumanNeanderthalDenisovan10",
            "Bonobo_100",
            "Bonobo_50",
            "Bonobo_10",
            "BonoboGhost",
            "HumanNeanderthal"]
    
    paths4 = ['afr100_GeneticMap_2',
            'afr100_GeneticMap']

    valid_loaders = []
    valid_loaders_small = []
    infos = []
    for path in paths4:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.valid_mixed + path + "_test%d_vcf_and_labels_test_random_" %i +str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.valid_ref_panel + path + "_test%d_vcf_and_labels_REF_random_" %i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                genetic_map="vcf/afr100_GeneticMap/geneticMap/plink.chr22.GRCh37.map",
                                                n_samples=32)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)

    for path in paths3:
        single_arc = 0
        for i in range(1, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.valid_mixed + path + "_test%d_vcf_and_labels_test_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.valid_ref_panel + path + "_test%d_vcf_and_labels_REF_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=10)

            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)

    for path in paths2:
        single_arc = 0
        for i in range(10, 11): #(2*rank+1,2*rank+3): #range(1 + 5 * rank, 6 + 5 * rank):
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.valid_mixed + path + "_test%d_vcf_and_labels_test_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_h5=args.valid_ref_panel + path + "_test%d_vcf_and_labels_REF_random_"%i+str(ratios2[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=1)

            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)

    for path in paths[:36]: #[paths[i] for i in [0,6,13,19,28,33,38,43,48,53]]: # paths[:60]: #  [paths[i] for i in [0, 5, 9, 12, 17, 21, 24, 30, 36, 42, 48, 54, 60, 61]]
        single_arc = 0
        if "DenSplitTime" in path:
            single_arc = 2
        elif "NeanSplitTime" in path:
            single_arc = 1
        for i in [1,2,3,4,5,10]:
            valid_dataset = ReferencePanelDatasetSmall(mixed_file_path=args.valid_mixed + path + "_test%d_vcf_and_labels_test_random_"%i+str(ratios[i-1])+".h5",
                                                reference_panel_h5=args.valid_ref_panel + path + "_test%d_vcf_and_labels_REF_random_"%i+str(ratios[i-1])+".h5",
                                                reference_panel_vcf=args.reference,
                                                reference_panel_map=args.map,
                                                n_refs_per_class=args.n_refs,
                                                transforms=transforms,
                                                single_arc=single_arc,
                                                n_samples=1)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
            infos.append(valid_dataset.info)
            valid_loaders.append(valid_loader)
            if path in paths[25:27]:
                valid_loaders_small.append(valid_loader)

    print("Data loaded")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    return valid_loaders, valid_loaders_small, infos


def smoother_validate(model, val_loaders, criterion, epoch, args):
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        val_loss = AverageMeter()
        acc = AverageMeter()
        recall = AverageMeter()
        precision = AverageMeter()
        f1 = AverageMeter()

        val_loss_per = AverageMeter()
        acc_per = AverageMeter()
        recall_per = AverageMeter()
        precision_per = AverageMeter()
        f1_per = AverageMeter()

        for idx_loader in range(len(val_loaders)):
            val_loader = val_loaders[idx_loader]

            val_loss_per.reset()
            acc_per.reset()
            recall_per.reset()
            precision_per.reset()
            f1_per.reset()

            for i, (preds, pos, lbl, single_arc) in enumerate(val_loader):
                preds = preds.to(device).float()  # (B,C,L)
                if pos is not None:
                    pos = pos.to(device).float()
                    pos = (pos / 1000000 / 100).to(preds.device) + torch.zeros([preds.shape[0],1,preds.shape[2]], dtype=torch.float32).to(preds.device)
                if lbl is not None:
                    lbl = lbl.to(device)

                out = model.forward(preds, pos=pos)

                if lbl is not None:
                    lbl[lbl==3] = 2  
                    if single_arc is not None:
                        if single_arc.item() == 1:
                            lbl[lbl == 2] = 1
                    lbl = lbl.unsqueeze(dim=1)
            
                if lbl is not None:
                    batch_acc, batch_recall, batch_precision, batch_f1 = ancestry_metrics(
                        out, lbl
                    )
                    acc.update(batch_acc.item())
                    recall.update(batch_recall.item())
                    precision.update(batch_precision.item())
                    f1.update(batch_f1.item())

                    acc_per.update(batch_acc.item())
                    recall_per.update(batch_recall.item())
                    precision_per.update(batch_precision.item())
                    f1_per.update(batch_f1.item())
                else:
                    pass

                if lbl is not None:
                    loss_val = criterion(out, lbl, epoch)
                    val_loss.update(loss_val.item())
                    val_loss_per.update(loss_val.item())

            print(idx_loader,
                  "acc:", acc_per.get_average(),
                  "recall:", recall_per.get_average(),
                  "precision:", precision_per.get_average(),
                  "f1:", f1_per.get_average(),
                  "loss:", val_loss_per.get_average()
            )
        model.train().to(device)
        return acc.get_average(), recall.get_average(), precision.get_average(), f1.get_average(), val_loss.get_average()


def validate(model, val_loaders, criterion, epoch, args):
    with torch.no_grad():
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.eval().to(device)
        # model = DataParallel(model, device_ids=[0, 1])

        val_loss = AverageMeter()
        acc = AverageMeter()
        recall = AverageMeter()
        precision = AverageMeter()
        f1 = AverageMeter()
        val_loss_per = AverageMeter()
        acc_per = AverageMeter()
        recall_per = AverageMeter()
        precision_per = AverageMeter()
        f1_per = AverageMeter()
        

        for _ in range(len(val_loaders)): # tqdm.tqdm(range(len(val_loaders))):
            val_loader = val_loaders[_]
            val_loss_per.reset()
            acc_per.reset()
            recall_per.reset()
            precision_per.reset()
            f1_per.reset()
            for i, batch in enumerate(val_loader):
                output = model.forward(batch, test=True)
                # label = f.pad(batch['mixed_labels'], (0, output['pad_num']), 'constant', 0)
                label = output["labels"]
                label[label==3]=2
                arc = batch["single_arc"][0]
                if arc==1:
                    label[label==2]=1
                label = label.unsqueeze(dim=1)
                
                # batch_acc, batch_recall, batch_precision, batch_f1 = ancestry_metrics(output["predictions"],
                #                                                                       batch["mixed_labels"].to(device))
                batch_acc, batch_recall, batch_precision, batch_f1 = ancestry_metrics(output["predictions"],
                                                                                      label.to(device))
                # batch_acc, batch_recall,batch_precision,batch_f1 = ancestry_metrics_ad(output)
                acc.update(batch_acc.item())
                recall.update(batch_recall.item())
                precision.update(batch_precision.item())
                f1.update(batch_f1.item())

                acc_per.update(batch_acc.item())
                recall_per.update(batch_recall.item())
                precision_per.update(batch_precision.item())
                f1_per.update(batch_f1.item())

                loss = criterion(output["out_basemodel"], label.to(device),epoch)
                val_loss.update(loss.item())
                val_loss_per.update(loss.item())
            print(_, "acc:", acc_per.get_average(), "recall:",recall_per.get_average(), "precision:",precision_per.get_average(), "f1:",f1_per.get_average(), "loss:",val_loss_per.get_average())

        return acc.get_average(), recall.get_average(), precision.get_average(), f1.get_average(), val_loss.get_average()


def inference(model, test_loader, args):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval().to(device)

        all_predictions = []

        for i, batch in enumerate(test_loader):
            batch = to_device(batch, device)

            output = model(batch, test=True, infer=True)

            probabilities = torch.softmax(output['predictions'], dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)

            predicted_labels = predicted_labels.reshape(batch['mixed_vcf'].shape[0],-1, 500)
            predicted_labels = predicted_labels.reshape(batch['mixed_vcf'].shape[0],-1)
            predicted_labels = predicted_labels[:, :-output["pad_num"]]
            all_predictions.append(predicted_labels)

        all_predictions = torch.cat(all_predictions, dim=0)

        return all_predictions


# ============================================================================
# RAW BED OUTPUT SUPPORT: SNP-level detail extraction functions
# These functions extract and save SNP-level information for each raw segment,
# enabling the merge script to perform exact merge replication
# ============================================================================

import gzip
import logging

def extract_snp_details_for_segments(seg_df, pos_array, hap_state_arrays, probabilities, hap_id_to_name_map):
    """
    Extract SNP-level details for each segment in seg_df

    Args:
        seg_df: DataFrame of segments (from find_introgression_segments)
        pos_array: numpy array of SNP positions
        hap_state_arrays: dict {haplotype_col: state_array}
        probabilities: [prob_label1, prob_label2]
        hap_id_to_name_map: dict mapping haplotype integer to sample_hap_id string

    Returns:
        List[dict]: SNP details for each segment
                   Using (chr, sample_hap_id, start_pos, end_pos) as composite key
    """
    if seg_df.empty:
        return []

    prob_label1 = probabilities[0][1]
    prob_label2 = probabilities[0][2]

    snp_details = []

    for idx, row in seg_df.iterrows():
        chr_val = row['chr']
        start_pos = row['start_pos']
        end_pos = row['end_pos']
        haplotype_col = row['haplotype']

        # Map haplotype integer to sample_hap_id string
        sample_hap_id = hap_id_to_name_map.get(haplotype_col, f'Unknown_HapID_{haplotype_col}')

        # Find SNP indices within this segment
        mask = (pos_array >= start_pos) & (pos_array <= end_pos)
        snp_indices = np.where(mask)[0]

        if len(snp_indices) == 0:
            logging.warning(f"Segment {sample_hap_id} at {start_pos}-{end_pos} has no SNPs")
            continue

        # Extract SNP information
        segment_positions = pos_array[snp_indices]
        segment_states = hap_state_arrays[haplotype_col][snp_indices]
        segment_prob1 = prob_label1[snp_indices]
        segment_prob2 = prob_label2[snp_indices]

        # Use position-based composite key instead of abstract segment_id
        # This allows merge script to match by position
        snp_details.append({
            'chr': chr_val,
            'sample_hap_id': sample_hap_id,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'snp_positions': ','.join(map(str, segment_positions.tolist())),
            'snp_states': ','.join(map(str, segment_states.tolist())),
            'snp_prob_label1': ','.join(f"{p:.6f}" for p in segment_prob1.tolist()),
            'snp_prob_label2': ','.join(f"{p:.6f}" for p in segment_prob2.tolist())
        })

    return snp_details


def write_snp_details_file(snp_details_list, filepath):
    """
    Write SNP details to gzip-compressed TSV file

    Args:
        snp_details_list: List[dict] of SNP details
        filepath: Output file path (.gz extension)
    """
    if not snp_details_list:
        logging.warning("No SNP details to write")
        return

    df = pd.DataFrame(snp_details_list)

    try:
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            df.to_csv(f, sep='\t', index=False, header=True)
        logging.info(f"Written {len(snp_details_list)} segment SNP details to {filepath}")
    except Exception as e:
        logging.error(f"Error writing SNP details file: {e}")


# Note: The following inference_and_write with simple signature is not used.
# The actual function with chunk processing parameters is defined below.
# This signature mismatch version is commented out to avoid confusion.

"""
def inference_and_write(base_model, smoother_model, test_loader, args, pred_file_path, bed_file_path, info):
    # This version is NOT USED - see the version with hap_id_to_name_map, is_first_chunk parameters below
    pass
"""

#ACTIVE VERSION: With raw.bed and SNP details output support
def inference_and_write(base_model, smoother_model, test_loader, args,
                        pred_file_path, bed_file_path, info,
                        hap_id_to_name_map,
                        is_first_chunk, current_index, current_index_filter):
    """
    RAW BED OUTPUT VERSION: Outputs both raw.bed and merged.bed files with SNP details

    This function:
    - Outputs TWO BED files:
      * *.raw.bed - Pre-merge segments (merge_distance=0)
      * *.bed - Merged segments (merge_distance=args.merge)
    - Outputs SNP details file (*.raw.snps.gz) for exact merge replication
    - Uses 10-column BED format (NO cM columns)
    - Maintains all memFast optimizations (batched writes, memory-efficient data types)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.eval().to(device)
    smoother_model.eval().to(device)

    chm = info['chm'][0]
    original_pos = [int(pos) for pos in info['pos']]
    original_pos_array = np.array(original_pos, dtype=int)

    max_pos = original_pos_array.max()
    pos_to_idx_array = -np.ones(max_pos + 1, dtype=int)
    pos_to_idx_array[original_pos_array] = np.arange(len(original_pos_array))

    # Batch results before writing to disk
    WRITE_BATCH_SIZE = 10
    pred_batch_buffer = []
    bed_batch_buffer_raw = []      # For raw.bed
    bed_batch_buffer_merged = []   # For merged.bed
    snp_details_buffer = []         # For raw.snps.gz

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader),
                            total=len(test_loader),
                            desc=f"Processing Chromosome {chm} (Chunk)",
                            unit="batch",
                            leave=False)

        for i, batch in progress_bar:
            batch = to_device(batch, device)

            basemodel_output = base_model(batch, test=True, infer=True)
            pad_num = basemodel_output.get("pad_num", 0)
            preds = basemodel_output["predictions"]
            pos = batch["pos"].to(preds.device)
            pos = (pos / 1000000 / 100).unsqueeze(1)

            output = smoother_model(preds, pos=pos)
            output = torch.nn.functional.pad(output, (0, pad_num), value=0)
            probabilities = torch.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            probabilities = probabilities.reshape(batch['mixed_vcf'].shape[0], 3, -1, 512).reshape(batch['mixed_vcf'].shape[0], 3, -1)
            if pad_num > 0:
                probabilities = probabilities[:, :, :-pad_num]

            predicted_labels = predicted_labels.reshape(batch['mixed_vcf'].shape[0], -1, 512).reshape(batch['mixed_vcf'].shape[0], -1)
            if pad_num > 0:
                predicted_labels = predicted_labels[:, :-pad_num]

            filtered_pos_batch = batch["pos"]

            # Transfer to CPU only once
            sample_label = predicted_labels[0].cpu().numpy().astype(int)
            sample_filtered_pos = filtered_pos_batch[0].cpu().numpy().astype(int)
            probabilities_np = probabilities.cpu().numpy().astype(np.float32)

            # Process pred.tsv data
            valid_mask = (sample_filtered_pos >= 0) & (sample_filtered_pos <= max_pos)
            valid_pos = sample_filtered_pos[valid_mask]
            valid_labels = sample_label[valid_mask]
            mapped_indices = pos_to_idx_array[valid_pos]
            if np.any(mapped_indices == -1):
                invalid_positions = valid_pos[mapped_indices == -1]
                raise KeyError(f"Invalid positions found: {invalid_positions.tolist()}")
            labels = np.zeros(len(original_pos_array), dtype=np.int8)
            labels[mapped_indices] = valid_labels

            pred_batch_buffer.append((current_index, labels))
            current_index += 1

            # Process bed file data
            a = predicted_labels.cpu().numpy().astype(np.int8)
            df_filtered = pd.DataFrame(a, columns=sample_filtered_pos)
            df_filtered.index = range(current_index_filter, current_index_filter + len(df_filtered))

            df_T = df_filtered.T.reset_index(drop=False)
            df_T.rename(columns={'index': 'POS'}, inplace=True)
            haplotype_columns = [col for col in df_T.columns if col != 'POS']

            # 1. Generate RAW segments (merge_distance=0)
            seg_df_raw = find_introgression_segments(
                df_T,
                haplotype_columns,
                probabilities_np,
                Chr=chm,
                merge_distance=0,  # No merge for raw output
                max_snp_gap_threshold=1_000_000,
                min_snps_per_segment=2,
                mosaic_minority_threshold=0.2
            )

            if not seg_df_raw.empty:
                # Extract SNP details for each raw segment
                pos_array = df_T['POS'].to_numpy(dtype=np.int64)
                hap_state_arrays = {col: df_T[col].to_numpy(dtype=np.int8) for col in haplotype_columns}
                batch_snp_details = extract_snp_details_for_segments(
                    seg_df_raw, pos_array, hap_state_arrays, probabilities_np, hap_id_to_name_map
                )
                snp_details_buffer.extend(batch_snp_details)

                # Prepare raw BED output (10 columns, NO cM)
                final_bed_df_raw = seg_df_raw.rename(columns={
                    'label': 'ancestry_label',
                    'snps': 'num_snps',
                    'prob': 'avg_prob',
                    'n_snps_label1': 'archaic_snps',
                    'n_snps_label2': 'african_snps'
                })
                final_bed_df_raw['sample_hap_id'] = final_bed_df_raw['haplotype'].map(hap_id_to_name_map)
                final_bed_df_raw['sample_hap_id'].fillna('Unknown_HapID', inplace=True)

                # 10 columns output (NO cM columns)
                output_columns_10 = [
                    'chr', 'start_pos', 'end_pos',
                    'haplotype', 'ancestry_label', 'num_snps', 'avg_prob',
                    'archaic_snps', 'african_snps', 'sample_hap_id'
                ]
                final_bed_df_raw = final_bed_df_raw[output_columns_10]
                final_bed_df_raw = final_bed_df_raw.sort_values(by=['sample_hap_id', 'chr', 'start_pos', 'end_pos'])
                bed_batch_buffer_raw.append(final_bed_df_raw)

            # 2. Generate MERGED segments (merge_distance=args.merge)
            seg_df_merged = find_introgression_segments(
                df_T,
                haplotype_columns,
                probabilities_np,
                Chr=chm,
                merge_distance=args.merge,
                max_snp_gap_threshold=1_000_000,
                min_snps_per_segment=2,
                mosaic_minority_threshold=0.2
            )

            if not seg_df_merged.empty:
                # Prepare merged BED output (10 columns, NO cM)
                final_bed_df_merged = seg_df_merged.rename(columns={
                    'label': 'ancestry_label',
                    'snps': 'num_snps',
                    'prob': 'avg_prob',
                    'n_snps_label1': 'archaic_snps',
                    'n_snps_label2': 'african_snps'
                })
                final_bed_df_merged['sample_hap_id'] = final_bed_df_merged['haplotype'].map(hap_id_to_name_map)
                final_bed_df_merged['sample_hap_id'].fillna('Unknown_HapID', inplace=True)

                # 10 columns output (NO cM columns)
                output_columns_10 = [
                    'chr', 'start_pos', 'end_pos',
                    'haplotype', 'ancestry_label', 'num_snps', 'avg_prob',
                    'archaic_snps', 'african_snps', 'sample_hap_id'
                ]
                final_bed_df_merged = final_bed_df_merged[output_columns_10]
                final_bed_df_merged = final_bed_df_merged.sort_values(by=['sample_hap_id', 'chr', 'start_pos', 'end_pos'])
                bed_batch_buffer_merged.append(final_bed_df_merged)

            current_index_filter += a.shape[0]

            # Write to disk every WRITE_BATCH_SIZE batches or at the end
            should_write = (i + 1) % WRITE_BATCH_SIZE == 0 or (i + 1) == len(test_loader)

            if should_write:
                # Write pred.tsv batched data
                if pred_batch_buffer:
                    batch_indices = [item[0] for item in pred_batch_buffer]
                    batch_labels = np.array([item[1] for item in pred_batch_buffer])
                    df = pd.DataFrame(batch_labels, columns=original_pos, index=batch_indices)

                    txt_mode = 'a'
                    write_header = False
                    if is_first_chunk and i < WRITE_BATCH_SIZE:
                        txt_mode = 'w'
                        write_header = True

                    df.to_csv(pred_file_path, sep="\t", mode=txt_mode, header=write_header, index=True)
                    pred_batch_buffer.clear()

                # Write raw.bed batched data
                if bed_batch_buffer_raw:
                    combined_bed_df_raw = pd.concat(bed_batch_buffer_raw, ignore_index=True)
                    raw_bed_path = bed_file_path.replace('.bed', '.raw.bed')
                    bed_mode = 'a'
                    if is_first_chunk and i < WRITE_BATCH_SIZE:
                        bed_mode = 'w'
                    combined_bed_df_raw.to_csv(raw_bed_path, sep='\t', mode=bed_mode, index=False, header=False)
                    bed_batch_buffer_raw.clear()

                # Write merged.bed batched data
                if bed_batch_buffer_merged:
                    combined_bed_df_merged = pd.concat(bed_batch_buffer_merged, ignore_index=True)
                    bed_mode = 'a'
                    if is_first_chunk and i < WRITE_BATCH_SIZE:
                        bed_mode = 'w'
                    combined_bed_df_merged.to_csv(bed_file_path, sep='\t', mode=bed_mode, index=False, header=False)
                    bed_batch_buffer_merged.clear()

    # Write SNP details file at the end of each chunk
    if snp_details_buffer:
        snp_details_path = bed_file_path.replace('.bed', '.raw.snps.gz')
        if is_first_chunk:
            # First chunk: write with header
            write_snp_details_file(snp_details_buffer, snp_details_path)
        else:
            # Subsequent chunks: append without header
            try:
                with gzip.open(snp_details_path, 'rt', encoding='utf-8') as f:
                    existing_df = pd.read_csv(f, sep='\t')
                new_df = pd.DataFrame(snp_details_buffer)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                with gzip.open(snp_details_path, 'wt', encoding='utf-8') as f:
                    combined_df.to_csv(f, sep='\t', index=False, header=True)
                logging.info(f"Appended {len(snp_details_buffer)} SNP details to {snp_details_path}")
            except Exception as e:
                logging.error(f"Error appending SNP details: {e}")

    return current_index, current_index_filter

"""
def inference_and_write(base_model, smoother_model, test_loader, args, 
                                      pred_file_path, bed_file_path, info, 
                                      hap_id_to_name_map,
                                      is_first_chunk, current_index, current_index_filter):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.eval().to(device)
    smoother_model.eval().to(device)
    
    chm = info['chm'][0]
    
    # STAGE 1 & 2: 数据收集与pred.tsv写入 (这部分逻辑保持不变)
    num_haplotypes_in_chunk = len(test_loader.dataset.mixed_vcf)
    bed_data_collectors = {
        'labels': [[] for _ in range(num_haplotypes_in_chunk)],
        'pos': [[] for _ in range(num_haplotypes_in_chunk)],
        'probs': [[] for _ in range(num_haplotypes_in_chunk)],
    }
    hap_offset_in_chunk = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), 
                            total=len(test_loader), 
                            desc=f"Processing Chromosome {chm} (Chunk)", 
                            unit="batch",
                            leave=False) 
        
        for i, batch in progress_bar:
            batch = to_device(batch, device)
            
            # --- 模型推理与数据准备 (与之前版本相同) ---
            basemodel_output = base_model(batch, test=True, infer=True)
            pad_num = basemodel_output.get("pad_num", 0)
            preds = basemodel_output["predictions"]
            pos_for_smoother = (batch["pos"].to(preds.device) / 1000000 / 100).to(preds.device) + torch.zeros([preds.shape[0], 1, preds.shape[2]], dtype=torch.float32).to(preds.device)
            output = smoother_model(preds, pos=pos_for_smoother)
            output = torch.nn.functional.pad(output, (0, pad_num), value=0)
            probabilities = torch.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            
            batch_size = batch['mixed_vcf'].shape[0]
            reshaped_probs = probabilities.reshape(batch_size, 3, -1, 512).reshape(batch_size, 3, -1)
            reshaped_labels = predicted_labels.reshape(batch_size, -1, 512).reshape(batch_size, -1)
            if pad_num > 0:
                reshaped_probs = reshaped_probs[:, :, :-pad_num]
                reshaped_labels = reshaped_labels[:, :-pad_num]
            
            # --- pred.tsv 写入逻辑 (保持不变) ---
            original_pos = [int(p) for p in info['pos']]
            original_pos_array = np.array(original_pos, dtype=int)
            max_pos = original_pos_array.max()
            pos_to_idx_array = -np.ones(max_pos + 1, dtype=int)
            pos_to_idx_array[original_pos_array] = np.arange(len(original_pos_array))
            sample_label_for_tsv = reshaped_labels[0].cpu().numpy().astype(int)
            sample_pos_for_tsv = batch["pos"][0].cpu().numpy().astype(int)
            valid_mask_tsv = (sample_pos_for_tsv >= 0) & (sample_pos_for_tsv <= max_pos)
            valid_pos_tsv = sample_pos_for_tsv[valid_mask_tsv]
            valid_labels_tsv = sample_label_for_tsv[valid_mask_tsv]
            mapped_indices_tsv = pos_to_idx_array[valid_pos_tsv]
            if np.any(mapped_indices_tsv == -1):
                 raise KeyError("Invalid positions found for pred.tsv generation.")
            labels_tsv = np.zeros(len(original_pos_array), dtype=int)
            labels_tsv[mapped_indices_tsv] = valid_labels_tsv
            df_tsv = pd.DataFrame([labels_tsv], columns=original_pos)
            df_tsv.index = range(current_index, current_index + len(df_tsv))
            txt_mode = 'a'
            if is_first_chunk and i == 0:
                txt_mode = 'w'
            df_tsv.to_csv(pred_file_path, sep="\t", mode=txt_mode, header=False, index=True)
            current_index += len(df_tsv)
            
            # --- 为 BED 文件聚合数据 (保持不变) ---
            for hap_idx_in_batch in range(batch_size):
                true_hap_idx = hap_offset_in_chunk
                bed_data_collectors['labels'][true_hap_idx].append(reshaped_labels[hap_idx_in_batch].cpu().numpy())
                bed_data_collectors['pos'][true_hap_idx].append(batch["pos"][hap_idx_in_batch].cpu().numpy())
                bed_data_collectors['probs'][true_hap_idx].append(reshaped_probs[hap_idx_in_batch].cpu().numpy().astype(float))
                hap_offset_in_chunk += 1

    # STAGE 3: BED 文件生成 (所有单倍型结果收集完毕后)
    all_segments_to_write = []
    for hap_idx in range(num_haplotypes_in_chunk):
        if not bed_data_collectors['labels'][hap_idx]:
            continue

        full_pred_labels = np.concatenate(bed_data_collectors['labels'][hap_idx])
        full_pred_pos = np.concatenate(bed_data_collectors['pos'][hap_idx])
        full_pred_probs = np.concatenate(bed_data_collectors['probs'][hap_idx], axis=1)

        valid_mask = full_pred_pos >= 0
        full_pred_labels = full_pred_labels[valid_mask]
        full_pred_pos = full_pred_pos[valid_mask]
        full_pred_probs = full_pred_probs[:, valid_mask]
        
        df_T = pd.DataFrame({hap_idx: full_pred_labels}, index=full_pred_pos)
        df_T.index.name = 'POS'
        df_T = df_T.reset_index()

        # 确保这里调用的是重构后的 find_introgression_segments_refactored
        introgression_segments_df = find_introgression_segments(
            df_T, 
            haplotype_columns=[hap_idx],
            probabilities=[full_pred_probs],
            Chr=chm, 
            merge_distance=args.merge,
            max_snp_gap_threshold=1_000_000,
            min_snps_per_segment=2,
            mosaic_minority_threshold=0.2
        )
        
        if not introgression_segments_df.empty:
            all_segments_to_write.append(introgression_segments_df)

    # STAGE 4: 最终处理与写入 (所有单倍型片段合并后)
    if all_segments_to_write:
        final_bed_df = pd.concat(all_segments_to_write, ignore_index=True)
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修正点 1: 排序逻辑 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 先按单倍型ID排序，再按起始位置排序，确保Hap 0的结果总在Hap 1之前
        final_bed_df = final_bed_df.sort_values(by=['haplotype', 'start_pos'])
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 核心修正点 1: 排序逻辑 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修正点 2: 添加SampleHapID列 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 'haplotype'列现在是数字ID(0, 1, ...)，我们用它来生成最终的名称ID
        final_bed_df['sample_hap_id'] = final_bed_df['haplotype'].apply(
            lambda hap_idx: hap_id_to_name_map.get(current_index_filter + hap_idx, f"unknown_hap_{current_index_filter + hap_idx}")
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 核心修正点 2: 添加SampleHapID列 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        bed_mode = 'a'
        if is_first_chunk:
            bed_mode = 'w'
        
        final_bed_df.to_csv(bed_file_path, sep='\t', mode=bed_mode, index=False, header=False)

    current_index_filter += num_haplotypes_in_chunk
    return current_index, current_index_filter
"""

"""
def inference_and_write(base_model, smoother_model, test_loader, args, 
                        pred_file_path, bed_file_path, info, 
                        hap_id_to_name_map,
                        is_first_chunk, current_index, current_index_filter):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.eval().to(device)
    smoother_model.eval().to(device)
    
    chm = info['chm'][0]
    num_haplotypes_in_chunk = len(test_loader.dataset.mixed_vcf)

    data_collectors = {
        'labels': [[] for _ in range(num_haplotypes_in_chunk)],
        'pos': [[] for _ in range(num_haplotypes_in_chunk)],
        'probs': [[] for _ in range(num_haplotypes_in_chunk)],
    }
    hap_offset_in_chunk = 0
    
    original_pos = [int(p) for p in info['pos']]
    original_pos_array = np.array(original_pos, dtype=int)
    max_pos = 0
    if len(original_pos_array) > 0: max_pos = original_pos_array.max()
    pos_to_idx_array = -np.ones(max_pos + 1, dtype=int)
    pos_to_idx_array[original_pos_array] = np.arange(len(original_pos_array))

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), 
                            desc=f"Processing Chromosome {chm} (Chunk)", unit="batch", leave=False) 
        
        for i, batch in progress_bar:
            batch = to_device(batch, device)
            basemodel_output = base_model(batch, test=True, infer=True)
            pad_num = basemodel_output.get("pad_num", 0)
            preds = basemodel_output["predictions"]
            pos_for_smoother = (batch["pos"].to(preds.device) / 1000000 / 100).to(preds.device) + torch.zeros([preds.shape[0], 1, preds.shape[2]], dtype=torch.float32).to(preds.device)
            output = smoother_model(preds, pos=pos_for_smoother)
            output = torch.nn.functional.pad(output, (0, pad_num), value=0)
            probabilities = torch.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            
            batch_size = batch['mixed_vcf'].shape[0]
            reshaped_probs = probabilities.reshape(batch_size, 3, -1, 512).reshape(batch_size, 3, -1)
            reshaped_labels = predicted_labels.reshape(batch_size, -1, 512).reshape(batch_size, -1)
            if pad_num > 0 and reshaped_labels.shape[1] > pad_num:
                reshaped_probs = reshaped_probs[:, :, :-pad_num]
                reshaped_labels = reshaped_labels[:, :-pad_num]

            for hap_idx_in_batch in range(batch_size):
                true_hap_idx = hap_offset_in_chunk
                data_collectors['labels'][true_hap_idx].append(reshaped_labels[hap_idx_in_batch].cpu().numpy())
                data_collectors['pos'][true_hap_idx].append(batch["pos"][hap_idx_in_batch].cpu().numpy())
                data_collectors['probs'][true_hap_idx].append(reshaped_probs[hap_idx_in_batch].cpu().numpy().astype(float))
                hap_offset_in_chunk += 1

    all_segments_to_write = []
    all_labels_for_tsv = []
    for hap_idx in range(num_haplotypes_in_chunk):
        if not data_collectors['labels'][hap_idx]: continue

        full_pred_labels = np.concatenate(data_collectors['labels'][hap_idx])
        full_pred_pos = np.concatenate(data_collectors['pos'][hap_idx])
        full_pred_probs = np.concatenate(data_collectors['probs'][hap_idx], axis=1)

        valid_mask = full_pred_pos >= 0
        full_pred_labels = full_pred_labels[valid_mask]
        full_pred_pos = full_pred_pos[valid_mask]
        full_pred_probs = full_pred_probs[:, valid_mask]
        
        mapped_indices = pos_to_idx_array[full_pred_pos]
        if np.any(mapped_indices == -1):
             raise KeyError(f"Haplotype {hap_idx}: Invalid positions found.")
        labels_for_one_hap = np.zeros(len(original_pos_array), dtype=int)
        labels_for_one_hap[mapped_indices] = full_pred_labels
        all_labels_for_tsv.append(labels_for_one_hap)

        df_T = pd.DataFrame({hap_idx: full_pred_labels}, index=full_pred_pos)
        df_T.index.name = 'POS'
        df_T = df_T.reset_index()
        
        introgression_segments_df = find_introgression_segments(df=df_T, haplotype_columns=[hap_idx], probabilities=[full_pred_probs], Chr=int(''.join(filter(str.isdigit, chm))), merge_distance=args.merge)
        
        if not introgression_segments_df.empty:
            all_segments_to_write.append(introgression_segments_df)

    if all_labels_for_tsv:
        hap_real_names = [hap_id_to_name_map.get(current_index_filter + i, f"unknown_hap_{i}") for i in range(len(all_labels_for_tsv))]
        df_pred = pd.DataFrame(all_labels_for_tsv, columns=original_pos, index=hap_real_names)
        txt_mode = 'a'; write_header = False
        if is_first_chunk: txt_mode = 'w'; write_header = True
        df_pred.to_csv(pred_file_path, sep="\t", mode=txt_mode, header=write_header, index=True)
        current_index += len(df_pred)

    if all_segments_to_write:
        final_bed_df = pd.concat(all_segments_to_write, ignore_index=True)
        final_bed_df = final_bed_df.sort_values(by=['haplotype', 'start_pos'])
        final_bed_df['sample_hap_id'] = final_bed_df['haplotype'].apply(lambda hap_idx: hap_id_to_name_map.get(current_index_filter + hap_idx, f"unknown_hap_{current_index_filter + hap_idx}"))
        bed_mode = 'a'
        if is_first_chunk: bed_mode = 'w'
        final_bed_df.to_csv(bed_file_path, sep='\t', mode=bed_mode, index=False, header=False)

    current_index_filter += num_haplotypes_in_chunk
    return current_index, current_index_filter
"""

class CustomBatchSampler(Sampler):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        # Create indices for all datasets
        self.indices = [range(len(d)) for d in datasets]

    def __iter__(self):
        # For each dataset
        for dataset_idx, dataset_indices in enumerate(self.indices):
            # Shuffle indices for this dataset
            shuffled_indices = torch.randperm(len(dataset_indices)).tolist()

            # Yield batches of indices
            for i in range(0, len(shuffled_indices), self.batch_size):
                batch_indices = shuffled_indices[i:i+self.batch_size]
                # Convert local dataset indices to global indices
                global_indices = [self.datasets[dataset_idx].indices[i] for i in batch_indices]
                yield global_indices

    def __len__(self):
        return sum(len(d) for d in self.datasets) // self.batch_size


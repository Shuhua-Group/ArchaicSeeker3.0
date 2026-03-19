import torch
import pickle
import numpy as np
from collections import Counter
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score
from torchvision import transforms


def ancestry_accuracy(prediction, target):
    b, l, c = prediction.shape
    prediction = prediction.reshape(b * l, c)
    target = target.reshape(b * l)

    prediction = prediction.max(dim=1)[1]
    accuracy = (prediction == target).sum()

    return accuracy / l



def filter_loci(batch):
    mixed_vcf = batch['mixed_vcf']
    mixed_labels = batch['mixed_labels']
    pos = batch['pos']
    ref_panel = batch['ref_panel']

    if isinstance(ref_panel, list) and all(isinstance(ref, dict) for ref in ref_panel):
        try:
            african_tensor = ref_panel[0][0].float()  
            den_tensor = ref_panel[0][1].float()      
            nean_tensor = ref_panel[0][2].float()     
        except Exception as e:
            print(f"Error when accessing ref_panel tensors: {e}")
            return batch  
    else:
        raise ValueError("Error")

    original_num_sites = mixed_vcf.shape[1]


    mask = torch.ones(mixed_vcf.shape[1], dtype=torch.bool, device=mixed_vcf.device)

    for idx in range(mixed_vcf.shape[1]):
        target_value = mixed_vcf[:, idx]
        african_value = african_tensor[:, idx].mean() if african_tensor.ndim > 1 else african_tensor[idx].float()
        den_value = den_tensor[:, idx].mean() if den_tensor.ndim > 1 else den_tensor[idx].float()
        nean_value = nean_tensor[:, idx].mean() if nean_tensor.ndim > 1 else nean_tensor[idx].float()
        if torch.all(target_value == african_value) and torch.all(target_value == den_value) and torch.all(target_value == nean_value):
            mask[idx] = False

    filtered_vcf = mixed_vcf[:, mask]
    filtered_labels = mixed_labels[:, mask]
    filtered_pos = pos[:, mask]
    filtered_num_sites = filtered_vcf.shape[1]

    batch['mixed_vcf'] = filtered_vcf
    batch['mixed_labels'] = filtered_labels
    batch['pos'] = filtered_pos

    for ref in ref_panel:
        try:
            ref[0] = ref[0][:, mask]  
            ref[1] = ref[1][:, mask]  
            ref[2] = ref[2][:, mask]  
        except Exception as e:
            print(f"Error when filtering ref_panel tensors: {e}")

    batch['ref_panel'] = ref_panel

    return batch


def ancestry_metrics(prediction, target, binary=False):
    prediction = prediction.permute(0,2,1)
    b, l, c = prediction.shape

    if binary:
        prediction = prediction.reshape(b*l)
        prediction = torch.floor(prediction + 0.5)
    else:
        prediction = prediction.reshape(b * l, c)
        prediction = prediction.max(dim=1)[1]
    target = target.reshape(b*l)

    
    accuracy = (prediction == target).sum()
    accuracy = accuracy / l /b
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()
    recall = recall_score(target, prediction, average='macro', zero_division=0)
    precision = precision_score(target, prediction, average='macro',zero_division=0)
    f1 = f1_score(target, prediction, average='macro', zero_division=0)
    return accuracy, recall, precision, f1


def ancestry_metrics_label_based(prediction, target, binary=False):
    prediction = prediction.view(-1)
    target = target.view(-1)

    if prediction.ndimension() == 1 and prediction.shape[0] == target.shape[0]:
        accuracy = (prediction == target).sum().float() / prediction.size(0)

        target = target.cpu().numpy()
        prediction = prediction.cpu().numpy()

        recall = recall_score(target, prediction, average='macro', zero_division=0)
        precision = precision_score(target, prediction, average='macro', zero_division=0)
        f1 = f1_score(target, prediction, average='macro', zero_division=0)

        return accuracy, recall, precision, f1
    else:
        raise ValueError("Predictions and targets have incompatible shapes.")



def ancestry_metrics_bin(prediction, target, binary=False):
    prediction = prediction.permute(0,2,1)
    b, l, c = prediction.shape

    if binary:
        prediction = prediction.reshape(b*l)
        prediction = torch.floor(prediction + 0.5)
    else:
        prediction = prediction.reshape(b * l, c)
        prediction = prediction.max(dim=1)[1]
    target = target.reshape(b*l)

    
    accuracy = (prediction == target).sum()
    accuracy = accuracy / l /b
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()
    recall = recall_score(target, prediction, zero_division=0)
    precision = precision_score(target, prediction,zero_division=0)
    f1 = f1_score(target, prediction, zero_division=0)
    return accuracy, recall, precision, f1

def ancestry_metrics_ad(output):
    b,l,s = output["predictions"].shape
    mse_neg = torch.mean(torch.pow(output["test"][output["train_indices"],0].reshape(-1,l*s) - output["test_predictions"][output['train_indices']].reshape(-1,l*s), 2), axis=1)
    mse_pos = torch.mean(torch.pow(output["test"][output["test_indices"],0].reshape(-1,l*s) - output["test_predictions"][output['test_indices']].reshape(-1,l*s), 2), axis=1)
    mean_neg = torch.mean(mse_neg)
    var_neg = torch.var(mse_neg)
    mean_pos = torch.mean(mse_pos)
    var_pos = torch.var(mse_pos)
    return mean_neg,var_neg,mean_pos,var_pos


class AverageMeter():
    def __init__(self):
        self.total = 0
        self.count = 0

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def get_average(self):
        return self.total / self.count


class ProgressSaver():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.progress = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_recall": [],
            "val_precision": [],
            "val_f1": [],
            "time": [],
            "best_epoch": [],
            "best_val_f1": [],
            "best_val_loss": [],
            "lr": [],
            "iter": []
        }

    def update_epoch_progess(self, epoch_data):
        for key in epoch_data.keys():
            self.progress[key].append(epoch_data[key])

        with open("%s/progress.pckl" % self.exp_dir, "wb") as f:
            pickle.dump(self.progress, f)

    def load_progress(self):
        with open("%s/progress.pckl" % self.exp_dir, "rb") as f:
            self.progress = pickle.load(f)

    def get_resume_stats(self):
        return self.progress["best_epoch"][-1], self.progress["best_val_loss"][-1], self.progress["iter"][-1], self.progress["time"][-1]


class ReshapedCrossEntropyLoss(nn.Module):
    def __init__(self, loss):
        super(ReshapedCrossEntropyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.Focal = FocalLoss()
        self.mse = nn.L1Loss()
        self.loss = loss

    def forward(self, prediction, target, epoch):
        if self.loss == "MSE":
            loss = self.mse(prediction, target[:,0,:].unsqueeze(1)) #target[:,0,:].unsqueeze(1)
            return loss
        prediction = prediction.permute(0,2,1)
        bs, seq_len, n_classes = prediction.shape
        prediction = prediction.reshape(bs * seq_len, n_classes)

        target = target.reshape(bs * seq_len)
        if self.loss == "CE":
            loss = self.CELoss(prediction, target)
        elif self.loss == "BCE":
            target = target.reshape(bs * seq_len, 1).to(torch.float)
            loss = self.BCELoss(prediction, target)
        elif self.loss == "LDAM":
            cls_num_list = []
            cls_num_list.append((target==0).sum().item())
            cls_num_list.append((target==1).sum().item())
            for i in range(len(cls_num_list)):
                if cls_num_list[i] == 0:
                    cls_num_list[i] == 1
            loss = LDAMLoss(cls_num_list, weight=None).forward(prediction, target)
        elif self.loss == "VS":
            target = target.reshape(bs * seq_len, 1).to(torch.float)
            cls_num_list = []
            cls_num_list.append((target==0).sum().item())
            cls_num_list.append((target==1).sum().item())
            loss = VSLoss(cls_num_list).forward(prediction, target)
        else:
            loss = self.Focal(prediction, target)
        return loss


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        target = target.type(torch.LongTensor).to(x.device)
        target = target.reshape(target.shape[0])
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :].to(index_float.device), index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.25, 0.25, 0.25], gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.nll_loss(torch.log(inputs), targets, reduction='none')

        at =  self.alpha.to(targets.device)[targets]

        # gather the specific corresponding `-log(pt)` for each target class
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.2, tau=1.2, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        target = target.type(torch.LongTensor).to(x.device)
        target = target.reshape(target.shape[0])
        output = x / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch / lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class EncodeBinary:

    def __call__(self, inp):
        # 0 -> -1
        # 1 -> 1
        inp["mixed_vcf"] = inp["mixed_vcf"] * 2 - 1
        for anc in inp["ref_panel"]:
            inp["ref_panel"][anc] = inp["ref_panel"][anc] * 2 - 1

        return inp


def build_transforms(args):
    transforms_list = []

    transforms_list.append(EncodeBinary())

    transforms_list = transforms.Compose(transforms_list)

    return transforms_list


def to_device(item, device):
    item["mixed_vcf"] = item["mixed_vcf"].to(device)

    if "mixed_labels" in item.keys():
        item["mixed_labels"] = item["mixed_labels"].to(device)

    for i, panel in enumerate(item["ref_panel"]):
        for anc in panel.keys():
            item["ref_panel"][i][anc] = item["ref_panel"][i][anc].to(device)

    return item


def correct_max_indices(max_indices_batch, ref_panel_idx_batch):
    '''
    for each element of a batch, the dataloader samples randomly a set of founders in random order. For this reason,
    the argmax values output by the base model will represent different associations of founders, depending on how they have been
    sampled and ordered. By storing the sampling information during the data loading, we can then correct the argmax outputs
    into a shared meaning between batches and elements within the batch.
    '''

    for n in range(len(max_indices_batch)):

        max_indices = max_indices_batch[n]
        ref_panel_idx = ref_panel_idx_batch[n]
        max_indices_ordered = [None] * len(ref_panel_idx.keys())

        for i, c in enumerate(ref_panel_idx.keys()):
            max_indices_ordered[c] = max_indices[i]
        max_indices_ordered = torch.stack(max_indices_ordered)

        for i in range(max_indices.shape[0]):
            max_indices_ordered[i] = torch.take(torch.tensor(ref_panel_idx[i]), max_indices_ordered[i].cpu())

        max_indices_batch[n] = max_indices_ordered[:]

    return max_indices_batch


def compute_ibd(output):
    all_ibd = []
    for n in range(output['out_basemodel'].shape[0]):
        classes_basemodel = torch.argmax(output['out_basemodel'][n], dim=0)
        # classes_smoother = torch.argmax(output['out_smoother'][n], dim=0)
        ibd = torch.gather(output['max_indices'][n].t(), index=classes_basemodel.unsqueeze(1), dim=1)
        ibd = ibd.squeeze(1)

        all_ibd.append(ibd)

    all_ibd = torch.stack(all_ibd)

    return all_ibd


def get_meta_data(chm, model_pos, query_pos, n_wind, wind_size, gen_map_df=None):
    """
    from LAI-Net code
    Transforms the predictions on a window level to a .msp file format.
        - chm: chromosome number
        - model_pos: physical positions of the model input SNPs in basepair units
        - query_pos: physical positions of the query input SNPs in basepair units
        - n_wind: number of windows in model
        - wind_size: size of each window in the model
        - genetic_map_file: the input genetic map file
    """

    model_chm_len = len(model_pos)

    # chm
    chm_array = [chm] * n_wind

    # start and end pyshical positions
    if model_chm_len % wind_size == 0:
        spos_idx = np.arange(0, model_chm_len, wind_size)  # [:-1]
        epos_idx = np.concatenate([np.arange(0, model_chm_len, wind_size)[1:], np.array([model_chm_len])]) - 1
    else:
        spos_idx = np.arange(0, model_chm_len, wind_size)[:-1]
        epos_idx = np.concatenate([np.arange(0, model_chm_len, wind_size)[1:-1], np.array([model_chm_len])]) - 1

    spos = model_pos[spos_idx]
    epos = model_pos[epos_idx]

    sgpos = [1] * len(spos)
    egpos = [1] * len(epos)

    # number of query snps in interval
    wind_index = [min(n_wind - 1, np.where(q == sorted(np.concatenate([epos, [q]])))[0][0]) for q in query_pos]
    window_count = Counter(wind_index)
    n_snps = [window_count[w] for w in range(n_wind)]

    # print(len(chm_array), len(spos), len(epos), len(sgpos), len(egpos), len(n_snps))
    # Concat with prediction table
    meta_data = np.array([chm_array, spos, epos, sgpos, egpos, n_snps]).T
    meta_data_df = pd.DataFrame(meta_data)
    meta_data_df.columns = ["chm", "spos", "epos", "sgpos", "egpos", "n snps"]

    return meta_data_df


def write_msp_tsv(output_folder, meta_data, pred_labels, populations, query_samples, write_population_code=False):
    msp_data = np.concatenate([np.array(meta_data), pred_labels.T], axis=1).astype(str)

    with open(output_folder + "/predictions.msp.tsv", 'w') as f:
        if write_population_code:
            # first line (comment)
            f.write("#Subpopulation order/codes: ")
            f.write("\t".join([str(pop) + "=" + str(i) for i, pop in enumerate(populations)]) + "\n")
        # second line (comment/header)
        f.write("#" + "\t".join(meta_data.columns) + "\t")
        f.write("\t".join([str(s) for s in np.concatenate([[s + ".0", s + ".1"] for s in query_samples])]) + "\n")
        # rest of the lines (data)
        for l in range(msp_data.shape[0]):
            f.write("\t".join(msp_data[l, :]))
            f.write("\n")

    return


def msp_to_lai(msp_file, positions, lai_file=None):
    msp_df = pd.read_csv(msp_file, sep="\t", comment="#", header=None)
    data_window = np.array(msp_df.iloc[:, 6:])
    n_reps = msp_df.iloc[:, 5].to_numpy()
    assert np.sum(n_reps) == len(positions)
    data_snp = np.concatenate([np.repeat([row], repeats=n_reps[i], axis=0) for i, row in enumerate(data_window)])

    with open(msp_file) as f:
        first_line = f.readline()
        second_line = f.readline()

    header = second_line[:-1].split("\t")
    samples = header[6:]
    df = pd.DataFrame(data_snp, columns=samples, index=positions)

    if lai_file is not None:
        with open(lai_file, "w") as f:
            f.write(first_line)
        df.to_csv(lai_file, sep="\t", mode='a', index_label="position")

def find_introgression_segments(
    df: pd.DataFrame,
    haplotype_columns: list,
    probabilities, 
    Chr: int = 1,
    merge_distance: int = 0,
    max_snp_gap_threshold: int = 1_000_000,
    min_snps_per_segment: int = 2,
    mosaic_minority_threshold: float = 0.20
) -> pd.DataFrame:
    """
    Identifies and merges archaic human introgression segments,
    implements splitting by large SNP gaps, and determines Mosaic types.
    Outputs numerical labels: 1 (Neanderthal-like), 2 (Denisova-like), 3 (Mosaic).
    Probability parameter handling strictly follows the original code's access pattern.
    """
    final_output_columns = ['chr', 'start_pos', 'end_pos', 'haplotype',
                            'label', 'snps', 'prob',
                            'n_snps_label1', 'n_snps_label2'] # Added component SNP counts

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=final_output_columns)

    if not all(col in df.columns for col in ['POS'] + haplotype_columns):
        raise ValueError("Input DataFrame is missing 'POS' column or specified haplotype_columns.")

    pos_array = df['POS'].to_numpy(dtype=np.int64)
    num_snps_total = len(pos_array) # For subsequent probability array length validation

    prob_arrays_for_archaic_labels = {}
    try:
        for label_val in [1, 2]: 
            raw_prob_data = probabilities[0][label_val]

            if hasattr(raw_prob_data, 'to_numpy'):
                prob_arrays_for_archaic_labels[label_val] = raw_prob_data.to_numpy(dtype=np.float64)
            elif isinstance(raw_prob_data, (np.ndarray, list)):
                prob_arrays_for_archaic_labels[label_val] = np.array(raw_prob_data, dtype=np.float64)
            else:
                raise TypeError(f"probabilities[0][{label_val}] (type: {type(raw_prob_data)}) is not the expected array type.")

            if len(prob_arrays_for_archaic_labels[label_val]) != num_snps_total:
                raise ValueError(f"Length of probabilities[0][{label_val}] ({len(prob_arrays_for_archaic_labels[label_val])}) does not match SNP data rows ({num_snps_total}).")
    except Exception as e:
        raise ValueError(f"Error processing probabilities parameter: {e}. Ensure `probabilities` structure allows `probabilities[0][1]` and `probabilities[0][2]` to return array/Series of same length as SNPs.")

    all_final_segments_for_haplotypes_list = []

    for hap_col in haplotype_columns:
        if hap_col not in df.columns:
            continue

        hap_state_array = df[hap_col].to_numpy(dtype=np.int8)

        is_archaic_snp = np.isin(hap_state_array, [1, 2])
        if not np.any(is_archaic_snp):
            continue

        padded_is_archaic = np.concatenate(([False], is_archaic_snp, [False]))
        diff_is_archaic = np.diff(padded_is_archaic.astype(np.int8))

        block_start_indices = np.where(diff_is_archaic == 1)[0]
        block_end_indices = np.where(diff_is_archaic == -1)[0] - 1

        if block_start_indices.size == 0:
            continue

        refined_sub_blocks_indices = []
        for s_idx, e_idx in zip(block_start_indices, block_end_indices):
            if (e_idx - s_idx + 1) < min_snps_per_segment:
                continue
            current_block_snp_indices = np.arange(s_idx, e_idx + 1)
            current_block_positions = pos_array[current_block_snp_indices]
            sub_block_start_abs_idx = s_idx
            if len(current_block_positions) <= 1:
                if (e_idx - sub_block_start_abs_idx + 1) >= min_snps_per_segment:
                    refined_sub_blocks_indices.append((sub_block_start_abs_idx, e_idx))
                continue
            gaps_in_block = np.diff(current_block_positions)
            split_after_indices_local = np.where(gaps_in_block > max_snp_gap_threshold)[0]
            for split_idx_local in split_after_indices_local:
                sub_block_end_abs_idx = current_block_snp_indices[split_idx_local]
                if (sub_block_end_abs_idx - sub_block_start_abs_idx + 1) >= min_snps_per_segment:
                    refined_sub_blocks_indices.append((sub_block_start_abs_idx, sub_block_end_abs_idx))
                sub_block_start_abs_idx = current_block_snp_indices[split_idx_local + 1]
            if (e_idx - sub_block_start_abs_idx + 1) >= min_snps_per_segment:
                refined_sub_blocks_indices.append((sub_block_start_abs_idx, e_idx))
        if not refined_sub_blocks_indices: continue

        classified_segments_this_hap_list_of_dicts = []
        for s_idx, e_idx in refined_sub_blocks_indices:
            segment_snp_states = hap_state_array[s_idx : e_idx + 1]
            n_label1_snps = np.sum(segment_snp_states == 1) # Count of Neanderthal-like SNPs
            n_label2_snps = np.sum(segment_snp_states == 2) # Count of Denisova-like SNPs
            total_archaic_snps_in_block = n_label1_snps + n_label2_snps

            if total_archaic_snps_in_block < min_snps_per_segment : # Redundant check if min_snps_per_segment applied earlier, but good for safety
                continue

            final_label_numeric = 0 # Default/unassigned
            if n_label1_snps > 0 and n_label2_snps > 0: # Both types of archaic SNPs present
                minority_count = min(n_label1_snps, n_label2_snps)
                if (minority_count / total_archaic_snps_in_block) >= mosaic_minority_threshold:
                    final_label_numeric = 3 # Mosaic
                else: # Not mosaic, assign the label of the dominant archaic type
                    final_label_numeric = 1 if n_label1_snps >= n_label2_snps else 2
            elif n_label1_snps > 0: # Only Neanderthal-like SNPs
                final_label_numeric = 1
            elif n_label2_snps > 0: # Only Denisova-like SNPs
                final_label_numeric = 2
            else: # No archaic SNPs (state 1 or 2) in this segment (should ideally not be reached)
                continue

            prob_values_for_segment_calc = []
            snp_indices_in_segment_calc = np.arange(s_idx, e_idx + 1)
            states_in_segment_for_prob = hap_state_array[snp_indices_in_segment_calc]

            if final_label_numeric == 1: # Neanderthal-like segment
                mask = (states_in_segment_for_prob == 1)
                if np.any(mask): prob_values_for_segment_calc = prob_arrays_for_archaic_labels[1][snp_indices_in_segment_calc[mask]]
            elif final_label_numeric == 2: # Denisova-like segment
                mask = (states_in_segment_for_prob == 2)
                if np.any(mask): prob_values_for_segment_calc = prob_arrays_for_archaic_labels[2][snp_indices_in_segment_calc[mask]]
            elif final_label_numeric == 3: # Mosaic segment
                # Probabilities are taken from the respective archaic ancestry arrays
                probs1_vals = prob_arrays_for_archaic_labels[1][snp_indices_in_segment_calc[states_in_segment_for_prob == 1]]
                probs2_vals = prob_arrays_for_archaic_labels[2][snp_indices_in_segment_calc[states_in_segment_for_prob == 2]]
                valid_probs1 = probs1_vals[~np.isnan(probs1_vals)] if len(probs1_vals) > 0 else np.array([])
                valid_probs2 = probs2_vals[~np.isnan(probs2_vals)] if len(probs2_vals) > 0 else np.array([])
                prob_values_for_segment_calc = np.concatenate((valid_probs1, valid_probs2))

            mean_segment_prob = np.mean(prob_values_for_segment_calc) if len(prob_values_for_segment_calc) > 0 else np.nan

            classified_segments_this_hap_list_of_dicts.append({
                'chr': Chr, 'start_pos': pos_array[s_idx], 'end_pos': pos_array[e_idx],
                'haplotype': hap_col, 'label': final_label_numeric, # Using numerical label
                'snps': total_archaic_snps_in_block, 'prob': mean_segment_prob,
                'n_snps_label1': n_label1_snps, 'n_snps_label2': n_label2_snps,
                '_s_idx_original_span': s_idx, '_e_idx_original_span': e_idx })
        if not classified_segments_this_hap_list_of_dicts: continue

        df_classified_hap = pd.DataFrame(classified_segments_this_hap_list_of_dicts)
        if df_classified_hap.empty: continue

        # Iterate through numerical labels [1 (Nea), 2 (Den), 3 (Mosaic)] for merging
        for numeric_label_to_merge in [1, 2, 3]:
            segments_this_label_group = df_classified_hap[df_classified_hap['label'] == numeric_label_to_merge].copy()
            if segments_this_label_group.empty: continue
            segments_this_label_group.sort_values(by='start_pos', inplace=True)
            # No need to check len(segments_this_label_group) == 0 again due to .empty check

            current_merged_seg_info = segments_this_label_group.iloc[0].to_dict()
            if not isinstance(current_merged_seg_info, dict) or '_s_idx_original_span' not in current_merged_seg_info:
                # print(f"Warning: Failed to initialize current_merged_seg_info (Label: {numeric_label_to_merge}, Haplotype: {hap_col})")
                continue

            for i in range(1, len(segments_this_label_group)):
                next_seg_info = segments_this_label_group.iloc[i].to_dict()
                gap = next_seg_info['start_pos'] - current_merged_seg_info['end_pos']
                if gap <= merge_distance:
                    current_merged_seg_info['end_pos'] = max(current_merged_seg_info['end_pos'], next_seg_info['end_pos'])
                    current_merged_seg_info['_e_idx_original_span'] = max(current_merged_seg_info['_e_idx_original_span'], next_seg_info['_e_idx_original_span'])
                else:
                    s_final_idx = current_merged_seg_info['_s_idx_original_span']
                    e_final_idx = current_merged_seg_info['_e_idx_original_span']
                    final_segment_snp_states = hap_state_array[s_final_idx : e_final_idx + 1]
                    final_n1 = np.sum(final_segment_snp_states == 1)
                    final_n2 = np.sum(final_segment_snp_states == 2)
                    final_total_snps_count = final_n1 + final_n2
                    if final_total_snps_count >= min_snps_per_segment :
                        final_prob_values_calc = []
                        indices_final_merge = np.arange(s_final_idx, e_final_idx + 1)
                        prob_label1_for_merge = prob_arrays_for_archaic_labels[1][indices_final_merge]
                        prob_label2_for_merge = prob_arrays_for_archaic_labels[2][indices_final_merge]

                        # current_merged_seg_info['label'] is already numeric_label_to_merge here
                        if current_merged_seg_info['label'] == 1: # Neanderthal-like
                            mask_final = (final_segment_snp_states == 1)
                            if np.any(mask_final): final_prob_values_calc = prob_label1_for_merge[mask_final]
                        elif current_merged_seg_info['label'] == 2: # Denisova-like
                            mask_final = (final_segment_snp_states == 2)
                            if np.any(mask_final): final_prob_values_calc = prob_label2_for_merge[mask_final]
                        elif current_merged_seg_info['label'] == 3: # Mosaic
                            p1_fm = prob_label1_for_merge[final_segment_snp_states == 1]
                            p2_fm = prob_label2_for_merge[final_segment_snp_states == 2]
                            valid_p1_fm = p1_fm[~np.isnan(p1_fm)] if len(p1_fm) > 0 else np.array([])
                            valid_p2_fm = p2_fm[~np.isnan(p2_fm)] if len(p2_fm) > 0 else np.array([])
                            final_prob_values_calc = np.concatenate((valid_p1_fm, valid_p2_fm))

                        final_mean_prob = np.mean(final_prob_values_calc) if len(final_prob_values_calc) > 0 else np.nan
                        all_final_segments_for_haplotypes_list.append(pd.DataFrame({
                            'chr': [Chr], 'start_pos': [current_merged_seg_info['start_pos']],
                            'end_pos': [current_merged_seg_info['end_pos']], 'haplotype': [hap_col],
                            'label': [current_merged_seg_info['label']], 'snps': [final_total_snps_count], # Label is numeric
                            'prob': [final_mean_prob], 'n_snps_label1': [final_n1], 'n_snps_label2': [final_n2]}))
                    current_merged_seg_info = next_seg_info.copy()

            # Process the last segment (or the only segment if loop didn't run)
            if current_merged_seg_info is not None and isinstance(current_merged_seg_info, dict) and '_s_idx_original_span' in current_merged_seg_info:
                s_final_idx = current_merged_seg_info['_s_idx_original_span']
                e_final_idx = current_merged_seg_info['_e_idx_original_span']
                final_segment_snp_states = hap_state_array[s_final_idx : e_final_idx + 1]
                final_n1 = np.sum(final_segment_snp_states == 1); final_n2 = np.sum(final_segment_snp_states == 2)
                final_total_snps_count = final_n1 + final_n2
                if final_total_snps_count >= min_snps_per_segment:
                    final_prob_values_calc = []
                    indices_final_merge = np.arange(s_final_idx, e_final_idx + 1)
                    prob_label1_for_merge = prob_arrays_for_archaic_labels[1][indices_final_merge]
                    prob_label2_for_merge = prob_arrays_for_archaic_labels[2][indices_final_merge]

                    if current_merged_seg_info['label'] == 1: # Neanderthal-like
                        mask_final = (final_segment_snp_states == 1)
                        if np.any(mask_final): final_prob_values_calc = prob_label1_for_merge[mask_final]
                    elif current_merged_seg_info['label'] == 2: # Denisova-like
                        mask_final = (final_segment_snp_states == 2)
                        if np.any(mask_final): final_prob_values_calc = prob_label2_for_merge[mask_final]
                    elif current_merged_seg_info['label'] == 3: # Mosaic
                        p1_fm = prob_label1_for_merge[final_segment_snp_states == 1]
                        p2_fm = prob_label2_for_merge[final_segment_snp_states == 2]
                        valid_p1_fm = p1_fm[~np.isnan(p1_fm)] if len(p1_fm) > 0 else np.array([])
                        valid_p2_fm = p2_fm[~np.isnan(p2_fm)] if len(p2_fm) > 0 else np.array([])
                        final_prob_values_calc = np.concatenate((valid_p1_fm, valid_p2_fm))

                    final_mean_prob = np.mean(final_prob_values_calc) if len(final_prob_values_calc) > 0 else np.nan
                    all_final_segments_for_haplotypes_list.append(pd.DataFrame({
                        'chr': [Chr], 'start_pos': [current_merged_seg_info['start_pos']],
                        'end_pos': [current_merged_seg_info['end_pos']], 'haplotype': [hap_col],
                        'label': [current_merged_seg_info['label']], 'snps': [final_total_snps_count], # Label is numeric
                        'prob': [final_mean_prob], 'n_snps_label1': [final_n1], 'n_snps_label2': [final_n2]}))

    if not all_final_segments_for_haplotypes_list:
        return pd.DataFrame(columns=final_output_columns)

    final_df_concat = pd.concat(all_final_segments_for_haplotypes_list, ignore_index=True)

    # Ensure correct dtypes
    dtype_map = {'chr': int, 'start_pos': np.int64, 'end_pos': np.int64,
                 'haplotype': object, 'label': pd.Int64Dtype(), # Changed label to Int64Dtype for numerical labels
                 'snps': pd.Int64Dtype(), 'prob': np.float64,
                 'n_snps_label1': pd.Int64Dtype(), 'n_snps_label2': pd.Int64Dtype()}

    for col, dtype_expected in dtype_map.items():
        if col in final_df_concat.columns:
            try:
                if final_df_concat[col].isnull().all():
                    if dtype_expected == pd.Int64Dtype(): final_df_concat[col] = pd.Series(dtype='Int64')
                    elif dtype_expected == np.float64: final_df_concat[col] = pd.Series(dtype='float64')
                    elif dtype_expected == object and final_df_concat[col].dtype == np.float64:
                         final_df_concat[col] = final_df_concat[col].astype(object)
                elif dtype_expected == pd.Int64Dtype():
                    final_df_concat[col] = final_df_concat[col].astype(pd.Int64Dtype())
                else:
                    final_df_concat[col] = final_df_concat[col].astype(dtype_expected)
            except Exception as e_astype:
                print(f"Warning: Could not astype column {col} to {str(dtype_expected)}. Current dtype: {final_df_concat[col].dtype}. Error: {e_astype}")

    return final_df_concat.reset_index(drop=True)

def find_introgression_segments(
    df: pd.DataFrame,
    haplotype_columns: list,
    probabilities,
    Chr: int = 1,
    merge_distance: int = 0,
    max_snp_gap_threshold: int = 1_000_000,
    min_snps_per_segment: int = 2,
    mosaic_minority_threshold: float = 0.20
) -> pd.DataFrame:
    """
    Identifies and merges archaic human introgression segments using a highly vectorized approach.
    This optimized version avoids explicit loops for segment splitting, classification, and merging,
    relying instead on efficient NumPy and pandas operations for significant speed improvements.

    Outputs numerical labels: 1 (Neanderthal-like), 2 (Denisova-like), 3 (Mosaic).
    """
    final_output_columns = [
        'chr', 'start_pos', 'end_pos', 'haplotype', 'label', 'snps', 'prob',
        'n_snps_label1', 'n_snps_label2'
    ]
    # --- 1. Initial Validation and Data Preparation ---
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=final_output_columns)

    if not all(col in df.columns for col in ['POS'] + haplotype_columns):
        raise ValueError("Input DataFrame is missing 'POS' column or specified haplotype_columns.")

    df.reset_index(drop=True, inplace=True) # Ensure default integer index
    pos_array = df['POS'].to_numpy(dtype=np.int64)
    num_snps_total = len(pos_array)

    try:
        prob1 = np.asarray(probabilities[0][1], dtype=np.float64)
        prob2 = np.asarray(probabilities[0][2], dtype=np.float64)
        if len(prob1) != num_snps_total or len(prob2) != num_snps_total:
            raise ValueError("Length of probability arrays does not match SNP data rows.")
    except Exception as e:
        raise ValueError(f"Error processing probabilities parameter: {e}.")

    all_haplotype_results = []

    # --- 2. Main Loop over Haplotypes ---
    for hap_col in haplotype_columns:
        if hap_col not in df.columns:
            continue

        hap_state_array = df[hap_col].to_numpy(dtype=np.int8)
        is_archaic_snp = np.isin(hap_state_array, [1, 2])
        if not np.any(is_archaic_snp):
            continue
        
        # Create a working DataFrame for archaic SNPs of the current haplotype
        archaic_indices = np.where(is_archaic_snp)[0]
        work_df = pd.DataFrame({
            'pos': pos_array[archaic_indices],
            'state': hap_state_array[archaic_indices],
            'prob1': prob1[archaic_indices],
            'prob2': prob2[archaic_indices],
        })
        
        # --- 3. Vectorized Segment Identification & Splitting ---
        # A new segment starts if the SNP is not contiguous or the gap is too large
        is_new_segment_start = np.diff(archaic_indices, prepend=-1) > 1
        gap_to_prev = np.diff(work_df['pos'], prepend=-1) > max_snp_gap_threshold
        
        # Combine conditions: a new segment ID is assigned at each start point
        segment_id_array = (is_new_segment_start | gap_to_prev).cumsum()
        work_df['segment_id'] = segment_id_array

        # --- 4. Vectorized Initial Segment Aggregation & Classification ---
        # Group by the new segment ID to get stats for each pre-merged segment
        grouped = work_df.groupby('segment_id')
        
        # Aggregate all necessary stats in one pass
        agg_funcs = {
            'pos': ['min', 'max', 'count'],
            'prob1': 'sum',
            'prob2': 'sum'
        }
        # Dynamically add state counting using a lambda
        agg_funcs['state'] = [
            ('n1', lambda s: (s == 1).sum()),
            ('n2', lambda s: (s == 2).sum())
        ]
        
        segments = grouped.agg(agg_funcs)
        segments.columns = ['start_pos', 'end_pos', 'snps', 'prob1_sum', 'prob2_sum', 'n_snps_label1', 'n_snps_label2']
        
        # Filter segments by minimum SNP count
        segments = segments[segments['snps'] >= min_snps_per_segment].copy()
        if segments.empty:
            continue

        # --- 5. Vectorized Label Assignment (including Mosaic) ---
        n1 = segments['n_snps_label1']
        n2 = segments['n_snps_label2']
        total_snps = n1 + n2
        
        # Conditions for each label
        is_mosaic = (n1 > 0) & (n2 > 0) & (np.minimum(n1, n2) / total_snps >= mosaic_minority_threshold)
        is_label1 = (~is_mosaic) & (n1 >= n2)
        is_label2 = (~is_mosaic) & (n2 > n1)
        
        # Assign labels using np.select for vectorization
        segments['label'] = np.select(
            [is_label1, is_label2, is_mosaic],
            [1, 2, 3],
            default=0
        )
        
        # --- 6. Vectorized Segment Merging ---
        segments.sort_values(by='start_pos', inplace=True)

        final_merged_segments = []
        # Process each label group separately for merging
        for label_val in [1, 2, 3]:
            label_group = segments[segments['label'] == label_val].copy()
            if label_group.empty:
                continue

            # Identify merge points in a vectorized way
            gap_to_next = label_group['start_pos'].shift(-1) - label_group['end_pos']
            # A new merge group starts where the gap is too large or it's the last segment
            is_new_merge_group = (gap_to_next > merge_distance) | gap_to_next.isna()
            label_group['merge_group_id'] = is_new_merge_group.cumsum()

            # Aggregate based on the merge group ID
            merged = label_group.groupby('merge_group_id').agg(
                start_pos=('start_pos', 'min'),
                end_pos=('end_pos', 'max'),
                n_snps_label1=('n_snps_label1', 'sum'),
                n_snps_label2=('n_snps_label2', 'sum'),
                prob1_sum=('prob1_sum', 'sum'),
                prob2_sum=('prob2_sum', 'sum'),
            )
            merged['label'] = label_val # Assign the correct label for the group
            final_merged_segments.append(merged)
        
        if not final_merged_segments:
            continue
            
        hap_df = pd.concat(final_merged_segments)
        
        # --- 7. Final Calculations & Formatting ---
        hap_df['snps'] = hap_df['n_snps_label1'] + hap_df['n_snps_label2']
        
        # Vectorized probability calculation
        prob_num = np.select(
            [hap_df['label'] == 1, hap_df['label'] == 2, hap_df['label'] == 3],
            [hap_df['prob1_sum'], hap_df['prob2_sum'], hap_df['prob1_sum'] + hap_df['prob2_sum']],
            default=0
        )
        prob_den = np.select(
             [hap_df['label'] == 1, hap_df['label'] == 2, hap_df['label'] == 3],
            [hap_df['n_snps_label1'], hap_df['n_snps_label2'], hap_df['snps']],
            default=1 # Avoid division by zero
        )
        
        # Use np.divide for safe division
        hap_df['prob'] = np.divide(prob_num, prob_den, out=np.full_like(prob_num, np.nan), where=prob_den!=0)

        hap_df['chr'] = Chr
        hap_df['haplotype'] = hap_col
        
        all_haplotype_results.append(hap_df[final_output_columns])

    # --- 8. Final Concatenation and Cleanup ---
    if not all_haplotype_results:
        return pd.DataFrame(columns=final_output_columns)

    final_df = pd.concat(all_haplotype_results, ignore_index=True).sort_values(
        by=['haplotype', 'start_pos']
    ).reset_index(drop=True)
    
    # Ensure correct dtypes for the final output
    dtype_map = {
        'chr': int, 'start_pos': np.int64, 'end_pos': np.int64,
        'haplotype': object, 'label': pd.Int64Dtype(),
        'snps': pd.Int64Dtype(), 'prob': np.float64,
        'n_snps_label1': pd.Int64Dtype(), 'n_snps_label2': pd.Int64Dtype()
    }
    return final_df.astype(dtype_map)

def find_introgression_segments(
    df: pd.DataFrame,
    haplotype_columns: list,
    probabilities,  # Directly used, assuming probabilities[0][label] is accessible
    Chr: int = 1,
    merge_distance: int = 0,
    # New parameters
    max_snp_gap_threshold: int = 1_000_000,
    min_snps_per_segment: int = 2,
    mosaic_minority_threshold: float = 0.20
) -> pd.DataFrame:
    """
    Identifies and merges archaic human introgression segments,
    implements splitting by large SNP gaps, and determines Mosaic types.
    Outputs numerical labels: 1 (Neanderthal-like), 2 (Denisova-like), 3 (Mosaic).
    Probability parameter handling strictly follows the original code's access pattern.
    """
    final_output_columns = ['chr', 'start_pos', 'end_pos', 'haplotype',
                            'label', 'snps', 'prob',
                            'n_snps_label1', 'n_snps_label2'] # Added component SNP counts

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=final_output_columns)

    if not all(col in df.columns for col in ['POS'] + haplotype_columns):
        raise ValueError("Input DataFrame is missing 'POS' column or specified haplotype_columns.")

    pos_array = df['POS'].to_numpy(dtype=np.int64)
    num_snps_total = len(pos_array) # For subsequent probability array length validation

    # Pre-extract and validate probability arrays for probabilities[0][1] and probabilities[0][2]
    # Strictly adhering to original access, with added length checks
    prob_arrays_for_archaic_labels = {}
    try:
        for label_val in [1, 2]: # Assuming archaic labels are 1 (e.g., Neanderthal) and 2 (e.g., Denisova)
            raw_prob_data = probabilities[0][label_val]

            if hasattr(raw_prob_data, 'to_numpy'):
                prob_arrays_for_archaic_labels[label_val] = raw_prob_data.to_numpy(dtype=np.float64)
            elif isinstance(raw_prob_data, (np.ndarray, list)):
                prob_arrays_for_archaic_labels[label_val] = np.array(raw_prob_data, dtype=np.float64)
            else:
                raise TypeError(f"probabilities[0][{label_val}] (type: {type(raw_prob_data)}) is not the expected array type.")

            if len(prob_arrays_for_archaic_labels[label_val]) != num_snps_total:
                raise ValueError(f"Length of probabilities[0][{label_val}] ({len(prob_arrays_for_archaic_labels[label_val])}) does not match SNP data rows ({num_snps_total}).")
    except Exception as e:
        raise ValueError(f"Error processing probabilities parameter: {e}. Ensure `probabilities` structure allows `probabilities[0][1]` and `probabilities[0][2]` to return array/Series of same length as SNPs.")

    all_final_segments_for_haplotypes_list = []
    # Numerical labels: 1 for Neanderthal-like, 2 for Denisova-like, 3 for Mosaic.
    # These correspond to how hap_state_array uses 1 and 2 for SNP types.

    for hap_col in haplotype_columns:
        if hap_col not in df.columns:
            # print(f"Warning: Haplotype column '{hap_col}' not in df, skipped.")
            continue

        hap_state_array = df[hap_col].to_numpy(dtype=np.int8)

        # Identify SNPs belonging to archaic types (states 1 or 2)
        is_archaic_snp = np.isin(hap_state_array, [1, 2])
        if not np.any(is_archaic_snp):
            continue

        padded_is_archaic = np.concatenate(([False], is_archaic_snp, [False]))
        diff_is_archaic = np.diff(padded_is_archaic.astype(np.int8))

        block_start_indices = np.where(diff_is_archaic == 1)[0]
        block_end_indices = np.where(diff_is_archaic == -1)[0] - 1

        if block_start_indices.size == 0:
            continue

        refined_sub_blocks_indices = []
        for s_idx, e_idx in zip(block_start_indices, block_end_indices):
            if (e_idx - s_idx + 1) < min_snps_per_segment:
                continue
            current_block_snp_indices = np.arange(s_idx, e_idx + 1)
            current_block_positions = pos_array[current_block_snp_indices]
            sub_block_start_abs_idx = s_idx
            if len(current_block_positions) <= 1:
                if (e_idx - sub_block_start_abs_idx + 1) >= min_snps_per_segment:
                    refined_sub_blocks_indices.append((sub_block_start_abs_idx, e_idx))
                continue
            gaps_in_block = np.diff(current_block_positions)
            split_after_indices_local = np.where(gaps_in_block > max_snp_gap_threshold)[0]
            for split_idx_local in split_after_indices_local:
                sub_block_end_abs_idx = current_block_snp_indices[split_idx_local]
                if (sub_block_end_abs_idx - sub_block_start_abs_idx + 1) >= min_snps_per_segment:
                    refined_sub_blocks_indices.append((sub_block_start_abs_idx, sub_block_end_abs_idx))
                sub_block_start_abs_idx = current_block_snp_indices[split_idx_local + 1]
            if (e_idx - sub_block_start_abs_idx + 1) >= min_snps_per_segment:
                refined_sub_blocks_indices.append((sub_block_start_abs_idx, e_idx))
        if not refined_sub_blocks_indices: continue

        classified_segments_this_hap_list_of_dicts = []
        for s_idx, e_idx in refined_sub_blocks_indices:
            segment_snp_states = hap_state_array[s_idx : e_idx + 1]
            n_label1_snps = np.sum(segment_snp_states == 1) # Count of Neanderthal-like SNPs
            n_label2_snps = np.sum(segment_snp_states == 2) # Count of Denisova-like SNPs
            total_archaic_snps_in_block = n_label1_snps + n_label2_snps

            if total_archaic_snps_in_block < min_snps_per_segment : # Redundant check if min_snps_per_segment applied earlier, but good for safety
                continue

            final_label_numeric = 0 # Default/unassigned
            if n_label1_snps > 0 and n_label2_snps > 0: # Both types of archaic SNPs present
                minority_count = min(n_label1_snps, n_label2_snps)
                if (minority_count / total_archaic_snps_in_block) >= mosaic_minority_threshold:
                    final_label_numeric = 3 # Mosaic
                else: # Not mosaic, assign the label of the dominant archaic type
                    final_label_numeric = 1 if n_label1_snps >= n_label2_snps else 2
            elif n_label1_snps > 0: # Only Neanderthal-like SNPs
                final_label_numeric = 1
            elif n_label2_snps > 0: # Only Denisova-like SNPs
                final_label_numeric = 2
            else: # No archaic SNPs (state 1 or 2) in this segment (should ideally not be reached)
                continue

            prob_values_for_segment_calc = []
            snp_indices_in_segment_calc = np.arange(s_idx, e_idx + 1)
            states_in_segment_for_prob = hap_state_array[snp_indices_in_segment_calc]

            if final_label_numeric == 1: # Neanderthal-like segment
                mask = (states_in_segment_for_prob == 1)
                if np.any(mask): prob_values_for_segment_calc = prob_arrays_for_archaic_labels[1][snp_indices_in_segment_calc[mask]]
            elif final_label_numeric == 2: # Denisova-like segment
                mask = (states_in_segment_for_prob == 2)
                if np.any(mask): prob_values_for_segment_calc = prob_arrays_for_archaic_labels[2][snp_indices_in_segment_calc[mask]]
            elif final_label_numeric == 3: # Mosaic segment
                # Probabilities are taken from the respective archaic ancestry arrays
                probs1_vals = prob_arrays_for_archaic_labels[1][snp_indices_in_segment_calc[states_in_segment_for_prob == 1]]
                probs2_vals = prob_arrays_for_archaic_labels[2][snp_indices_in_segment_calc[states_in_segment_for_prob == 2]]
                valid_probs1 = probs1_vals[~np.isnan(probs1_vals)] if len(probs1_vals) > 0 else np.array([])
                valid_probs2 = probs2_vals[~np.isnan(probs2_vals)] if len(probs2_vals) > 0 else np.array([])
                prob_values_for_segment_calc = np.concatenate((valid_probs1, valid_probs2))

            mean_segment_prob = np.mean(prob_values_for_segment_calc) if len(prob_values_for_segment_calc) > 0 else np.nan

            classified_segments_this_hap_list_of_dicts.append({
                'chr': Chr, 'start_pos': pos_array[s_idx], 'end_pos': pos_array[e_idx],
                'haplotype': hap_col, 'label': final_label_numeric, # Using numerical label
                'snps': total_archaic_snps_in_block, 'prob': mean_segment_prob,
                'n_snps_label1': n_label1_snps, 'n_snps_label2': n_label2_snps,
                '_s_idx_original_span': s_idx, '_e_idx_original_span': e_idx })
        if not classified_segments_this_hap_list_of_dicts: continue

        df_classified_hap = pd.DataFrame(classified_segments_this_hap_list_of_dicts)
        if df_classified_hap.empty: continue

        # Iterate through numerical labels [1 (Nea), 2 (Den), 3 (Mosaic)] for merging
        for numeric_label_to_merge in [1, 2, 3]:
            segments_this_label_group = df_classified_hap[df_classified_hap['label'] == numeric_label_to_merge].copy()
            if segments_this_label_group.empty: continue
            segments_this_label_group.sort_values(by='start_pos', inplace=True)
            # No need to check len(segments_this_label_group) == 0 again due to .empty check

            current_merged_seg_info = segments_this_label_group.iloc[0].to_dict()
            if not isinstance(current_merged_seg_info, dict) or '_s_idx_original_span' not in current_merged_seg_info:
                # print(f"Warning: Failed to initialize current_merged_seg_info (Label: {numeric_label_to_merge}, Haplotype: {hap_col})")
                continue

            for i in range(1, len(segments_this_label_group)):
                next_seg_info = segments_this_label_group.iloc[i].to_dict()
                gap = next_seg_info['start_pos'] - current_merged_seg_info['end_pos']
                if gap <= merge_distance:
                    current_merged_seg_info['end_pos'] = max(current_merged_seg_info['end_pos'], next_seg_info['end_pos'])
                    current_merged_seg_info['_e_idx_original_span'] = max(current_merged_seg_info['_e_idx_original_span'], next_seg_info['_e_idx_original_span'])
                else:
                    s_final_idx = current_merged_seg_info['_s_idx_original_span']
                    e_final_idx = current_merged_seg_info['_e_idx_original_span']
                    final_segment_snp_states = hap_state_array[s_final_idx : e_final_idx + 1]
                    final_n1 = np.sum(final_segment_snp_states == 1)
                    final_n2 = np.sum(final_segment_snp_states == 2)
                    final_total_snps_count = final_n1 + final_n2
                    if final_total_snps_count >= min_snps_per_segment :
                        final_prob_values_calc = []
                        indices_final_merge = np.arange(s_final_idx, e_final_idx + 1)
                        prob_label1_for_merge = prob_arrays_for_archaic_labels[1][indices_final_merge]
                        prob_label2_for_merge = prob_arrays_for_archaic_labels[2][indices_final_merge]

                        # current_merged_seg_info['label'] is already numeric_label_to_merge here
                        if current_merged_seg_info['label'] == 1: # Neanderthal-like
                            mask_final = (final_segment_snp_states == 1)
                            if np.any(mask_final): final_prob_values_calc = prob_label1_for_merge[mask_final]
                        elif current_merged_seg_info['label'] == 2: # Denisova-like
                            mask_final = (final_segment_snp_states == 2)
                            if np.any(mask_final): final_prob_values_calc = prob_label2_for_merge[mask_final]
                        elif current_merged_seg_info['label'] == 3: # Mosaic
                            p1_fm = prob_label1_for_merge[final_segment_snp_states == 1]
                            p2_fm = prob_label2_for_merge[final_segment_snp_states == 2]
                            valid_p1_fm = p1_fm[~np.isnan(p1_fm)] if len(p1_fm) > 0 else np.array([])
                            valid_p2_fm = p2_fm[~np.isnan(p2_fm)] if len(p2_fm) > 0 else np.array([])
                            final_prob_values_calc = np.concatenate((valid_p1_fm, valid_p2_fm))

                        final_mean_prob = np.mean(final_prob_values_calc) if len(final_prob_values_calc) > 0 else np.nan
                        all_final_segments_for_haplotypes_list.append(pd.DataFrame({
                            'chr': [Chr], 'start_pos': [current_merged_seg_info['start_pos']],
                            'end_pos': [current_merged_seg_info['end_pos']], 'haplotype': [hap_col],
                            'label': [current_merged_seg_info['label']], 'snps': [final_total_snps_count], # Label is numeric
                            'prob': [final_mean_prob], 'n_snps_label1': [final_n1], 'n_snps_label2': [final_n2]}))
                    current_merged_seg_info = next_seg_info.copy()

            # Process the last segment (or the only segment if loop didn't run)
            if current_merged_seg_info is not None and isinstance(current_merged_seg_info, dict) and '_s_idx_original_span' in current_merged_seg_info:
                s_final_idx = current_merged_seg_info['_s_idx_original_span']
                e_final_idx = current_merged_seg_info['_e_idx_original_span']
                final_segment_snp_states = hap_state_array[s_final_idx : e_final_idx + 1]
                final_n1 = np.sum(final_segment_snp_states == 1); final_n2 = np.sum(final_segment_snp_states == 2)
                final_total_snps_count = final_n1 + final_n2
                if final_total_snps_count >= min_snps_per_segment:
                    final_prob_values_calc = []
                    indices_final_merge = np.arange(s_final_idx, e_final_idx + 1)
                    prob_label1_for_merge = prob_arrays_for_archaic_labels[1][indices_final_merge]
                    prob_label2_for_merge = prob_arrays_for_archaic_labels[2][indices_final_merge]

                    if current_merged_seg_info['label'] == 1: # Neanderthal-like
                        mask_final = (final_segment_snp_states == 1)
                        if np.any(mask_final): final_prob_values_calc = prob_label1_for_merge[mask_final]
                    elif current_merged_seg_info['label'] == 2: # Denisova-like
                        mask_final = (final_segment_snp_states == 2)
                        if np.any(mask_final): final_prob_values_calc = prob_label2_for_merge[mask_final]
                    elif current_merged_seg_info['label'] == 3: # Mosaic
                        p1_fm = prob_label1_for_merge[final_segment_snp_states == 1]
                        p2_fm = prob_label2_for_merge[final_segment_snp_states == 2]
                        valid_p1_fm = p1_fm[~np.isnan(p1_fm)] if len(p1_fm) > 0 else np.array([])
                        valid_p2_fm = p2_fm[~np.isnan(p2_fm)] if len(p2_fm) > 0 else np.array([])
                        final_prob_values_calc = np.concatenate((valid_p1_fm, valid_p2_fm))

                    final_mean_prob = np.mean(final_prob_values_calc) if len(final_prob_values_calc) > 0 else np.nan
                    all_final_segments_for_haplotypes_list.append(pd.DataFrame({
                        'chr': [Chr], 'start_pos': [current_merged_seg_info['start_pos']],
                        'end_pos': [current_merged_seg_info['end_pos']], 'haplotype': [hap_col],
                        'label': [current_merged_seg_info['label']], 'snps': [final_total_snps_count], # Label is numeric
                        'prob': [final_mean_prob], 'n_snps_label1': [final_n1], 'n_snps_label2': [final_n2]}))

    if not all_final_segments_for_haplotypes_list:
        return pd.DataFrame(columns=final_output_columns)

    final_df_concat = pd.concat(all_final_segments_for_haplotypes_list, ignore_index=True)

    # Ensure correct dtypes
    dtype_map = {'chr': int, 'start_pos': np.int64, 'end_pos': np.int64,
                 'haplotype': object, 'label': pd.Int64Dtype(), # Changed label to Int64Dtype for numerical labels
                 'snps': pd.Int64Dtype(), 'prob': np.float64,
                 'n_snps_label1': pd.Int64Dtype(), 'n_snps_label2': pd.Int64Dtype()}

    for col, dtype_expected in dtype_map.items():
        if col in final_df_concat.columns:
            try:
                if final_df_concat[col].isnull().all():
                    if dtype_expected == pd.Int64Dtype(): final_df_concat[col] = pd.Series(dtype='Int64')
                    elif dtype_expected == np.float64: final_df_concat[col] = pd.Series(dtype='float64')
                    elif dtype_expected == object and final_df_concat[col].dtype == np.float64:
                         final_df_concat[col] = final_df_concat[col].astype(object)
                elif dtype_expected == pd.Int64Dtype():
                    final_df_concat[col] = final_df_concat[col].astype(pd.Int64Dtype())
                else:
                    final_df_concat[col] = final_df_concat[col].astype(dtype_expected)
            except Exception as e_astype:
                print(f"Warning: Could not astype column {col} to {str(dtype_expected)}. Current dtype: {final_df_concat[col].dtype}. Error: {e_astype}")

    return final_df_concat.reset_index(drop=True)

# Latest no overlap
def find_introgression_segments(
    df: pd.DataFrame,
    haplotype_columns: list,
    probabilities,
    Chr: int = 1,
    merge_distance: int = 0,
    max_snp_gap_threshold: int = 1_000_000,
    min_snps_per_segment: int = 2,
    mosaic_minority_threshold: float = 0.20
) -> pd.DataFrame:
    """
    Definitive, refactored function to identify and merge archaic introgression segments.
    This version uses a single-pass, ordered merging strategy to prevent any
    possibility of nested or overlapping segment outputs.
    """

    final_output_columns = [
        'chr', 'start_pos', 'end_pos', 'haplotype', 'label', 'snps',
        'prob', 'n_snps_label1', 'n_snps_label2'
    ]

    if not isinstance(df, pd.DataFrame) or df.empty or not all(c in df.columns for c in ['POS'] + haplotype_columns):
        return pd.DataFrame(columns=final_output_columns)

    pos_array = df['POS'].to_numpy(dtype=np.int64)
    num_snps_total = len(pos_array)

    try:
        prob_label1 = np.array(probabilities[0][1], dtype=np.float64)
        prob_label2 = np.array(probabilities[0][2], dtype=np.float64)
        if len(prob_label1) != num_snps_total or len(prob_label2) != num_snps_total:
            raise ValueError("Probability array lengths do not match SNP data length.")
    except Exception as e:
        raise ValueError(f"Error processing probabilities: {e}")

    all_haplotypes_final_segments = []

    for hap_col in haplotype_columns:
        hap_state_array = df[hap_col].to_numpy(dtype=np.int8)
        is_archaic_snp = np.isin(hap_state_array, [1, 2])

        if not np.any(is_archaic_snp):
            continue

        # --- 1. Find initial candidate blocks (non-overlapping) ---
        padded = np.concatenate(([False], is_archaic_snp, [False]))
        diffs = np.diff(padded.astype(np.int8))
        block_starts = np.where(diffs == 1)[0]
        block_ends = np.where(diffs == -1)[0] - 1

        candidate_blocks = []
        for s_idx, e_idx in zip(block_starts, block_ends):
            sub_block_start = s_idx
            positions_in_block = pos_array[s_idx:e_idx + 1]
            gaps = np.diff(positions_in_block)
            split_points = np.where(gaps > max_snp_gap_threshold)[0]
            
            for split_idx in split_points:
                sub_block_end = s_idx + split_idx
                if (sub_block_end - sub_block_start + 1) >= min_snps_per_segment:
                    candidate_blocks.append((sub_block_start, sub_block_end))
                sub_block_start = s_idx + split_idx + 1
            
            if (e_idx - sub_block_start + 1) >= min_snps_per_segment:
                candidate_blocks.append((sub_block_start, e_idx))
        
        if not candidate_blocks:
            continue

        # --- 2. Create a DataFrame of initial, classified, non-overlapping segments ---
        initial_segments = []
        for s_idx, e_idx in candidate_blocks:
            initial_segments.append({
                'start_pos': pos_array[s_idx], 'end_pos': pos_array[e_idx],
                '_s_idx': s_idx, '_e_idx': e_idx
            })
        
        if not initial_segments: continue
        df_initial = pd.DataFrame(initial_segments).sort_values('start_pos')

        # --- 3. NEW: Single-pass, ordered merging logic ---
        if df_initial.empty: continue

        final_merged_segments = []
        # Start with the first segment as the current one to merge into
        current_seg = df_initial.iloc[0].to_dict()

        for i in range(1, len(df_initial)):
            next_seg = df_initial.iloc[i].to_dict()
            gap = next_seg['start_pos'] - current_seg['end_pos']

            if gap <= merge_distance:
                # If gap is small, merge next_seg into current_seg
                current_seg['end_pos'] = max(current_seg['end_pos'], next_seg['end_pos'])
                current_seg['_e_idx'] = max(current_seg['_e_idx'], next_seg['_e_idx'])
            else:
                # If gap is too large, the current merged segment is final. Add it.
                final_merged_segments.append(current_seg)
                # The next segment becomes the new "current" segment
                current_seg = next_seg
        
        # Add the very last segment after the loop finishes
        final_merged_segments.append(current_seg)
        
        # --- 4. Final classification and stat calculation on the TRUE merged blocks ---
        final_output_rows = []
        for seg in final_merged_segments:
            s_idx, e_idx = seg['_s_idx'], seg['_e_idx']
            
            segment_states = hap_state_array[s_idx : e_idx + 1]
            n1 = np.sum(segment_states == 1)
            n2 = np.sum(segment_states == 2)
            total_snps = n1 + n2

            if total_snps < min_snps_per_segment: continue

            # Classify the final, merged segment
            label = 0
            if n1 > 0 and n2 > 0:
                minority_ratio = min(n1, n2) / total_snps
                if minority_ratio >= mosaic_minority_threshold:
                    label = 3
                else:
                    label = 1 if n1 >= n2 else 2
            elif n1 > 0: label = 1
            elif n2 > 0: label = 2
            
            if label == 0: continue

            # Calculate final probability
            prob_values = []
            indices = np.arange(s_idx, e_idx + 1)
            if label == 1:
                prob_values = prob_label1[indices[segment_states == 1]]
            elif label == 2:
                prob_values = prob_label2[indices[segment_states == 2]]
            elif label == 3:
                prob_values = np.concatenate((
                    prob_label1[indices[segment_states == 1]],
                    prob_label2[indices[segment_states == 2]]
                ))
            
            mean_prob = np.mean(prob_values) if len(prob_values) > 0 else np.nan

            final_output_rows.append({
                'chr': Chr, 'start_pos': seg['start_pos'], 'end_pos': seg['end_pos'],
                'haplotype': hap_col, 'label': label, 'snps': total_snps, 
                'prob': mean_prob, 'n_snps_label1': n1, 'n_snps_label2': n2
            })
        
        if final_output_rows:
            all_haplotypes_final_segments.append(pd.DataFrame(final_output_rows))

    if not all_haplotypes_final_segments:
        return pd.DataFrame(columns=final_output_columns)

    final_df = pd.concat(all_haplotypes_final_segments, ignore_index=True)
    return final_df[final_output_columns]

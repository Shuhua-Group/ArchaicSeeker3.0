import pybedtools
import numpy as np
import os

def _calc_accuracy(truth_bt, infer_bt):
    truth_bt  = truth_bt.sort().merge()
    infer_bt  = infer_bt.sort().merge()

    total_inf  = sum(iv.length for iv in infer_bt)
    total_true = sum(iv.length for iv in truth_bt)
    tp_len     = sum(iv.length for iv in infer_bt.intersect(truth_bt))

    precision = np.nan if total_inf  == 0 else tp_len / total_inf  * 100
    recall    = np.nan if total_true == 0 else tp_len / total_true * 100

    f1 = np.nan if (
        np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0
    ) else 2 * precision * recall / (precision + recall)

    # ⭐ 新增指标：预测长度/真实长度
    len_ratio = np.nan if total_true == 0 else total_inf / total_true

    return precision, recall, f1, len_ratio



def _is_empty(path):
    """判断 bed 文件是否为空（无有效行）"""
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        return True
    # 检查是否至少有一行非空内容
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                return False
    return True


def cal_accuracy_hap(true_tracts_path: str,
                     inferred_tracts_path: str,
                     sample_col: int = 3):
    if _is_empty(true_tracts_path) or _is_empty(inferred_tracts_path):
        return np.nan, np.nan, np.nan, np.nan

    truth_all  = pybedtools.BedTool(true_tracts_path)
    infer_all  = pybedtools.BedTool(inferred_tracts_path)

    # 获取同时存在于真值和预测中的样本集合
    samples_truth = {iv.fields[sample_col] for iv in truth_all}
    samples_pred  = {iv.fields[sample_col] for iv in infer_all}
    samples = sorted(samples_truth & samples_pred)
    if not samples:
        return np.nan, np.nan, np.nan, np.nan

    results = {}
    for samp in samples:
        t_bt = truth_all.filter(lambda x: x.fields[sample_col] == samp).saveas()
        i_bt = infer_all.filter(lambda x: x.fields[sample_col] == samp).saveas()
        results[samp] = _calc_accuracy(t_bt, i_bt)

    arr = np.array(list(results.values()), dtype=float)
    if arr.size == 0:
        mean_prec = mean_rec = mean_f1 = np.nan
    else:
        mean_prec, mean_rec, mean_f1, mean_len_ratio = np.nanmean(arr, axis=0)


    return mean_prec, mean_rec, mean_f1, mean_len_ratio


def cal_accuracy_sample(true_tracts_path: str,
                        inferred_tracts_path: str,
                        sample_col: int = 3):
    if _is_empty(true_tracts_path) or _is_empty(inferred_tracts_path):
        return np.nan, np.nan, np.nan, np.nan

    truth_all  = pybedtools.BedTool(true_tracts_path)
    infer_all  = pybedtools.BedTool(inferred_tracts_path)

    def hap_to_sample_id(iv):
        hap_id = int(iv.fields[sample_col])
        return str(hap_id // 2)

    samples_truth = {hap_to_sample_id(iv) for iv in truth_all}
    samples_pred  = {hap_to_sample_id(iv) for iv in infer_all}
    samples = sorted(samples_truth & samples_pred)
    if not samples:
        return np.nan, np.nan, np.nan, np.nan

    results = {}
    for samp in samples:
        t_bt = truth_all.filter(lambda x: str(int(x.fields[sample_col]) // 2) == samp).saveas()
        i_bt = infer_all.filter(lambda x: str(int(x.fields[sample_col]) // 2) == samp).saveas()
        results[samp] = _calc_accuracy(t_bt, i_bt)

    arr = np.array(list(results.values()), dtype=float)
    mean_prec, mean_rec, mean_f1, mean_len_ratio = np.nanmean(arr, axis=0)

    return mean_prec, mean_rec, mean_f1, mean_len_ratio


def cal_accuracy_region(true_tracts_path: str, inferred_tracts_path: str):
    if _is_empty(true_tracts_path) or _is_empty(inferred_tracts_path):
        return np.nan, np.nan, np.nan, np.nan
    truth_bt   = pybedtools.BedTool(true_tracts_path)
    infer_bt   = pybedtools.BedTool(inferred_tracts_path)
    return _calc_accuracy(truth_bt, infer_bt)

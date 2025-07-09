import numpy as np
import os
import pickle
import hashlib  # 用于生成唯一的缓存文件名
import torch
from torch.utils.data import Dataset, DataLoader
import random
import h5py
import pandas as pd
import allel
import logging
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import pysam

class SmootherDataset(Dataset):
    """
    从 basemodel_inference.pt 中读取 [predictions, positions, labels]。
    先 merge_all=True => cat到一个大张量(1, C, L_total)，再以 chunk_size 切片。
    最后 squeeze 多余维度 => (1,C,chunkLen).
    """

    def __init__(self, pt_file, merge_all=True, chunk_size=50000):
        data = torch.load(pt_file, map_location='cpu')
        self.pred_list = data["predictions"]  # list of Tensors(1,C,L_i) or (B,C,L_i)
        self.pos_list  = data["positions"]
        self.lbl_list  = data["labels"]

        self.chunk_size = chunk_size
        self.merged = merge_all

        if self.merged:
            # merge all
            pred_cat = []
            pos_cat  = []
            lbl_cat  = []

            for p, pos, lbl in zip(self.pred_list, self.pos_list, self.lbl_list):
                # p => shape (1,C,L_i), or maybe (1,1,3,L_i)
                p = self._ensure_3d(p)

                pred_cat.append(p)   # => (1,C,L_i)

                if pos is not None:
                    pos = self._ensure_2d(pos)  # => (1,L_i)
                    pos_cat.append(pos)

                if lbl is not None:
                    lbl = self._ensure_2d(lbl)  # => (1,L_i)
                    lbl_cat.append(lbl)

            self.pred_merged = torch.cat(pred_cat, dim=2)  # => (1,C, sum_Li)
            if len(pos_cat)>0:
                self.pos_merged  = torch.cat(pos_cat, dim=1)  # => (1, sum_Li)
            else:
                self.pos_merged  = None

            if len(lbl_cat) == len(self.lbl_list):
                self.lbl_merged  = torch.cat(lbl_cat, dim=1) # =>(1, sum_Li)
            else:
                self.lbl_merged  = None

            self.total_length = self.pred_merged.shape[-1]
            self.n_chunks = (self.total_length + self.chunk_size - 1)// self.chunk_size
        else:
            # 不 merge => 逐个
            self.pred_merged = None
            self.pos_merged  = None
            self.lbl_merged  = None
            self.n_chunks = len(self.pred_list)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        if self.merged:
            # chunk模式
            start = idx*self.chunk_size
            end = min(start+self.chunk_size, self.total_length)

            chunk_pred = self.pred_merged[..., start:end]  # shape (1,C, chunkLen)
            chunk_pos  = None
            chunk_lbl  = None

            if self.pos_merged is not None:
                chunk_pos = self.pos_merged[..., start:end]  # (1, chunkLen)
            if self.lbl_merged is not None:
                chunk_lbl = self.lbl_merged[..., start:end]  # (1, chunkLen)

            # 保证 chunk_pred => 3D
            chunk_pred = self._ensure_3d(chunk_pred)  # => (1,C,chunkLen)
            if chunk_pos is not None:
                chunk_pos  = self._ensure_2d(chunk_pos)      # => (1,chunkLen)
            if chunk_lbl is not None:
                chunk_lbl  = self._ensure_2d(chunk_lbl)

            return chunk_pred, chunk_pos, chunk_lbl
        else:
            # 不merge => 直接取
            p = self.pred_list[idx]
            pos= self.pos_list[idx]
            lbl= self.lbl_list[idx]

            p   = self._ensure_3d(p)
            if pos is not None:
                pos= self._ensure_2d(pos)
            if lbl is not None:
                lbl= self._ensure_2d(lbl)
            return p, pos, lbl


    def _ensure_3d(self, t):
        """
        将 t squeeze 成 (B,C,L) 3D格式。
        若出现 (1,1,3,L) => (1,3,L)
        若是 (C,L) => =>(1,C,L)
        """
        while t.dim()>3:
            # 如果第二个维度=1, squeeze它
            # or 你也可写 if t.size(0)==1, t=t.squeeze(0), depends
            # but we want final shape = (B,C,L)
            # common case: (1,1,3,L)->(1,3,L)
            dims = list(t.shape)
            # find a dimension=1 => squeeze
            squeezed = False
            for d in range(len(dims)):
                if dims[d]==1 and t.dim()>3:
                    t = t.squeeze(d)
                    squeezed=True
                    break
            if not squeezed:
                break

        if t.dim()==2:
            # =>(C,L) => (1,C,L)
            t = t.unsqueeze(0)
        return t

    def _ensure_2d(self, t):
        """
        将 t squeeze 成 (B,L) 2D格式。
        e.g. (1,1,L)->(1,L), or (L)->(1,L)
        """
        while t.dim()>2:
            dims = list(t.shape)
            squeezed=False
            for d in range(len(dims)):
                if dims[d]==1 and t.dim()>2:
                    t = t.squeeze(d)
                    squeezed=True
                    break
            if not squeezed:
                break

        if t.dim()==1:
            t = t.unsqueeze(0)  # =>(1,L)
        return t

class SmootherDataset(Dataset):
    """
    从 basemodel_inference.pt 读取 predictions/positions/labels,
    不再做 chunk_size 拆分。
    如果 merge_all=True，就把所有 (1,C,L_i) 合并成 (1,C,L_total)，
    整个数据集就只有 1 条记录 => __len__=1。
    如果 merge_all=False，就直接保留多条 => len(self.pred_list)。
    """
    def __init__(self, pt_file, merge_all=True):
        data = torch.load(pt_file, map_location='cpu')
        self.pred_list = data["predictions"]  # list of Tensors(1,C,L_i)
        self.pos_list  = data["positions"]
        self.lbl_list  = data["labels"]

        self.merged = merge_all

        if self.merged:
            # 把所有 batch 合并
            pred_cat = []
            pos_cat  = []
            lbl_cat  = []
            for p, pos, lbl in zip(self.pred_list, self.pos_list, self.lbl_list):
                p   = self._ensure_3d(p)  # => (1,C,L_i)
                pred_cat.append(p)
                if pos is not None:
                    pos_cat.append( self._ensure_2d(pos) )  # =>(1,L_i)
                if lbl is not None:
                    lbl_cat.append( self._ensure_2d(lbl) )  # =>(1,L_i)

            self.pred_merged = torch.cat(pred_cat, dim=2)  # => (1,C,L_total)
            self.pos_merged  = torch.cat(pos_cat, dim=1) if len(pos_cat) else None
            if len(lbl_cat)==len(self.lbl_list):
                self.lbl_merged = torch.cat(lbl_cat, dim=1)
            else:
                self.lbl_merged = None

            # 仅有1条记录
            self.length = 1
        else:
            self.pred_merged = None
            self.pos_merged  = None
            self.lbl_merged  = None
            self.length = len(self.pred_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.merged:
            # 只返回同一个 (1,C,L_total)
            p = self.pred_merged
            pos= self.pos_merged
            lbl= self.lbl_merged
        else:
            p   = self._ensure_3d(self.pred_list[idx])
            pos = self.pos_list[idx]
            if pos is not None:
                pos = self._ensure_2d(pos)
            lbl = self.lbl_list[idx]
            if lbl is not None:
                lbl = self._ensure_2d(lbl)

        return p, pos, lbl
    
    def _ensure_3d(self, t):
        """
        将 t squeeze 成 (B,C,L) 3D格式。
        若出现 (1,1,3,L) => (1,3,L)
        若是 (C,L) => =>(1,C,L)
        """
        while t.dim()>3:
            # 如果第二个维度=1, squeeze它
            # or 你也可写 if t.size(0)==1, t=t.squeeze(0), depends
            # but we want final shape = (B,C,L)
            # common case: (1,1,3,L)->(1,3,L)
            dims = list(t.shape)
            # find a dimension=1 => squeeze
            squeezed = False
            for d in range(len(dims)):
                if dims[d]==1 and t.dim()>3:
                    t = t.squeeze(d)
                    squeezed=True
                    break
            if not squeezed:
                break

        if t.dim()==2:
            # =>(C,L) => (1,C,L)
            t = t.unsqueeze(0)
        return t

    def _ensure_2d(self, t):
        """
        将 t squeeze 成 (B,L) 2D格式。
        e.g. (1,1,L)->(1,L), or (L)->(1,L)
        """
        while t.dim()>2:
            dims = list(t.shape)
            squeezed=False
            for d in range(len(dims)):
                if dims[d]==1 and t.dim()>2:
                    t = t.squeeze(d)
                    squeezed=True
                    break
            if not squeezed:
                break

        if t.dim()==1:
            t = t.unsqueeze(0)  # =>(1,L)
        return t


def smoother_collate_fn(batch):
    """
    batch是一个list，长度=batch_size。
    其中每个元素都是 (pred, pos, lbl)，
    - pred => shape (1,3,chunkLen)
    - pos  => shape (1,chunkLen) or None
    - lbl  => shape (1,chunkLen) or None

    我们要把它们合并成:
      preds => (batch_size, 3, chunkLen)
      pos   => (batch_size, chunkLen)
      lbl   => (batch_size, chunkLen)

    若 batch_size=1，就只是一项 => 变成 (1,3,chunkLen).
    """
    preds_list = []
    pos_list   = []
    lbl_list   = []

    for (p, pos, lbl) in batch:
        # p => shape (1,3,chunkLen)
        preds_list.append(p)  # 维度 (1,3,chunkLen)
        pos_list.append(pos)  # (1,chunkLen) or None
        lbl_list.append(lbl)  # (1,chunkLen) or None

    # 把 preds_list 里的若干 (1,3,chunkLen) 拼接 => (batch_size, 3, chunkLen)
    preds_t = torch.cat(preds_list, dim=0)  # 在dim=0拼 => (B,3,chunkLen)

    # 同理 pos (若都不为 None)
    if all(x is not None for x in pos_list):
        pos_t = torch.cat(pos_list, dim=0)  # => (B, chunkLen)
    else:
        pos_t = None

    # 同理 lbl
    if all(x is not None for x in lbl_list):
        lbl_t = torch.cat(lbl_list, dim=0)  # => (B, chunkLen)
    else:
        lbl_t = None

    return preds_t, pos_t, lbl_t



def read_map(map_file):
    return pd.read_csv(map_file, sep='\t', header=None, names=['chr', 'id', 'gen_dist', 'position'])

def calculate_genetic_distances(map_df, positions):
    sorted_positions = np.sort(positions)
    indices = np.searchsorted(map_df['position'], sorted_positions, side='right') - 1
    lower_idx = np.maximum(0, indices)
    upper_idx = np.minimum(len(map_df) - 1, indices + 1)
    
    lower_positions = map_df.iloc[lower_idx]['position'].values
    upper_positions = map_df.iloc[upper_idx]['position'].values
    lower_distances = map_df.iloc[lower_idx]['gen_dist'].values
    upper_distances = map_df.iloc[upper_idx]['gen_dist'].values

    position_differences = upper_positions - lower_positions
    distance_differences = upper_distances - lower_distances
    
    slopes = np.zeros_like(position_differences)
    
    valid = position_differences != 0
    
    slopes[valid] = distance_differences[valid] / position_differences[valid]
    
    interpolated_distances = lower_distances + slopes * (sorted_positions - lower_positions)
    
    return interpolated_distances * 1000000

def to_tensor(item):
    for k in item.keys():
        item[k] = torch.tensor(item[k])

    item["vcf"] = item["vcf"].float()
    item["labels"] = item["labels"].long()
    return item

class GenomeDataset(Dataset):
    def __init__(self, data, transforms):
        data = np.load(data)
        self.vcf_data = data["vcf"].astype(np.float)
        self.labels = data["labels"]
        self.transforms = transforms

    def __len__(self):
        return self.vcf_data.shape[0]

    def __getitem__(self, item):
        item = {
            "vcf": self.vcf_data[item],
            "labels": self.labels[item]
        }

        item = to_tensor(item)
        item = self.transforms(item)
        return item

def load_refpanel_from_h5py(reference_panel_h5):
    reference_panel_file = h5py.File(reference_panel_h5, "r")
    return reference_panel_file["vcf"], reference_panel_file["labels"], reference_panel_file["pos"]

def load_map_file(map_file):
    sample_map = pd.read_csv(map_file, sep="\t", header=None)
    sample_map.columns = ["sample", "ancestry"]
    ancestry_names, ancestry_labels = np.unique(sample_map['ancestry'], return_inverse=True)
    samples_list = np.array(sample_map['sample'])
    return samples_list, ancestry_labels, ancestry_names

def load_vcf_samples_in_map(vcf_file, samples_list):
    # Reading VCF
    vcf_data = allel.read_vcf(vcf_file)

    # Intersection between samples from VCF and samples from .map
    inter = np.intersect1d(vcf_data['samples'], samples_list, assume_unique=False, return_indices=True)
    samp, idx = inter[0], inter[1]

    # Filter only intersecting samples
    snps = vcf_data['calldata/GT'].transpose(1, 2, 0)[idx, ...]
    samples = vcf_data['samples'][idx]

    # Save header info of VCF file
    info = {
        'chm': vcf_data['variants/CHROM'],
        'pos': vcf_data['variants/POS'],
        'id': vcf_data['variants/ID'],
        'ref': vcf_data['variants/REF'],
        'alt': vcf_data['variants/ALT'],
    }

    return samples, snps, info

def load_refpanel_from_vcfmap(reference_panel_vcf, reference_panel_samplemap):
    samples_list, ancestry_labels, ancestry_names = load_map_file(reference_panel_samplemap)
    samples_vcf, snps, info = load_vcf_samples_in_map(reference_panel_vcf, samples_list)

    argidx = np.argsort(samples_vcf)
    samples_vcf = samples_vcf[argidx]
    snps = snps[argidx, ...]

    argidx = np.argsort(samples_list)
    samples_list = samples_list[argidx]

    ancestry_labels = ancestry_labels[argidx, ...]

    # 扩展母系和父系序列
    ancestry_labels = np.expand_dims(ancestry_labels, axis=1)
    ancestry_labels = np.repeat(ancestry_labels, 2, axis=1)
    ancestry_labels = ancestry_labels.reshape(-1)

    samples_list_upsampled = []
    for sample_id in samples_list:
        for _ in range(2): samples_list_upsampled.append(sample_id)

    snps = snps.reshape(snps.shape[0] * 2, -1)

    return snps, ancestry_labels, samples_list_upsampled, ancestry_names, info

def vcf_to_npy(vcf_file):
    vcf_data = allel.read_vcf(vcf_file, fields=['samples', 'calldata/GT', 'variants/POS'])
    snps = vcf_data['calldata/GT'].transpose(1, 2, 0)
    samples = vcf_data['samples']
    positions = vcf_data['variants/POS']
    return snps, samples, positions


def ref_pan_to_tensor(item):
    item["mixed_vcf"] = torch.tensor(item["mixed_vcf"]).float()

    if "mixed_labels" in item.keys():
        item["mixed_labels"] = torch.tensor(item["mixed_labels"]).long()

    for c in item["ref_panel"]:
        item["ref_panel"][c] = torch.tensor(item["ref_panel"][c]).float()

    return item

class ReferencePanel:

    def __init__(self, reference_panel_vcf, reference_panel_labels, n_refs_per_class, samples_list=None, cache_dir="cache/"):

        self.reference_vcf = reference_panel_vcf
        self.samples_list = samples_list
        self.n_refs_per_class = n_refs_per_class

        # 创建缓存目录
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        reference_labels = reference_panel_labels
        reference_panel = {}

        for i, ref in enumerate(reference_labels):
            ancestry = np.unique(ref)
            # For now, it is only supported single ancestry references
            assert len(ancestry) == 1
            ancestry = int(ancestry)

            if ancestry in reference_panel.keys():
                reference_panel[ancestry].append(i)
            else:
                reference_panel[ancestry] = [i]

        self.reference_panel_index_dict = reference_panel
        # 保存所有的祖先类别键
        self.reference_keys = list(reference_panel.keys())

    def sample_uniform_all_classes(self, n_sample_per_class, filtered_ref_panel):
        if filtered_ref_panel is None:
            filtered_ref_panel = self.reference_vcf
        reference_samples = {}
        reference_samples_names = {}
        reference_samples_idx = {}
        for ancestry in self.reference_panel_index_dict.keys():
            n_samples = min(n_sample_per_class, len(self.reference_panel_index_dict[ancestry]))
            indexes = random.sample(self.reference_panel_index_dict[ancestry],
                                    n_samples)
            reference_samples_idx[ancestry] = indexes
            reference_samples[ancestry] = []
            reference_samples_names[ancestry] = []
            for i in indexes:
                reference_samples[ancestry].append(filtered_ref_panel[i])
                if self.samples_list is not None:
                    reference_samples_names[ancestry].append(self.samples_list[i])
                else:
                    reference_samples_names[ancestry].append(None)
            reference_samples = {x: np.array(reference_samples[x]) for x in
                                 reference_samples.keys()}

        return reference_samples, reference_samples_names, reference_samples_idx

    def sample_reference_panel(self, filtered_ref_panel = None):
        return self.sample_uniform_all_classes(n_sample_per_class=self.n_refs_per_class, filtered_ref_panel=filtered_ref_panel)

class ReferencePanelDataset(Dataset):
    """
    一个经过重构的数据集类，具备以下新特性：
    1. 自动处理Reference和Target VCF之间的位点交集。
    2. 支持按块加载Target样本，以显著降低内存消耗。
    """
    def __init__(self, mixed_file_path, samples_to_load, reference_panel_vcf, reference_panel_map, 
                 n_refs_per_class, transforms, single_arc=0, genetic_map=None, labels=None):

        logging.info("Initializing new Dataset chunk...")
        self.mixed_file_path = mixed_file_path
        self.samples_to_load = samples_to_load
        
        # --- 1. 加载Reference Panel的完整信息 ---
        # 这一步会加载所有参考样本和它们的位点，我们假设参考面板大小是可控的。
        logging.info("Loading reference panel...")
        ref_snps, ref_labels, ref_samples, _, ref_info = load_refpanel_from_vcfmap(
            reference_panel_vcf, reference_panel_map
        )
        ref_pos = ref_info['pos'].astype(np.int32)

        # --- 2. 仅加载Target VCF的位点信息 ---
        # 使用pysam高效地遍历VCF，只读取POS列，不加载基因型数据，内存占用极小。
        logging.info("Reading positions from target VCF...")
        with pysam.VariantFile(self.mixed_file_path) as vcf:
            target_pos = np.array([rec.pos for rec in vcf.fetch()], dtype=np.int32)

        # --- 3. 核心步骤：寻找共有位点并创建对齐索引 ---
        logging.info("Finding intersection of SNPs between reference and target...")
        common_positions, ref_indices, target_indices = np.intersect1d(
            ref_pos, target_pos, return_indices=True
        )
        if len(common_positions) == 0:
            raise ValueError("No common SNPs found between the reference panel and the target VCF. Please check your input files.")
        logging.info(f"Found {len(common_positions)} common SNPs. Aligning data...")

        # --- 4. 根据交集筛选(对齐)Reference Panel数据 ---
        ref_snps_aligned = ref_snps[:, ref_indices]
        
        # --- 5. 核心步骤：按需加载并对齐当前块的Target VCF数据 ---
        # 我们只加载 self.samples_to_load 中指定的样本
        logging.info(f"Loading and aligning genotype data for {len(self.samples_to_load)} target samples...")
        snps_chunk, _, pos_chunk = vcf_to_npy(self.mixed_file_path) # 复用旧函数加载数据
        
        # a. 筛选出共有的位点
        target_snps_aligned = snps_chunk[:, :, target_indices]
        
        # b. 筛选出当前块所需要的样本
        all_samples_in_vcf = list(_) # `_` 接收了vcf_to_npy返回的样本名列表
        sample_indices_in_vcf = [all_samples_in_vcf.index(s) for s in self.samples_to_load]
        
        # c. 从对齐后的数据中提取出这些样本的基因型
        target_snps_chunk_aligned = target_snps_aligned[sample_indices_in_vcf, :, :]
        
        # d. 将数据格式从 (n_samples, 2, n_snps) 转换为 (n_haplotypes, n_snps)
        n_seq, n_chann, n_snps = target_snps_chunk_aligned.shape
        self.mixed_vcf = target_snps_chunk_aligned.reshape(n_seq * n_chann, n_snps)
        self.mixed_pos = common_positions # 最终使用的位点是已对齐的共有位点

        # --- 6. 设置其他属性 ---
        self.reference_panel = ReferencePanel(ref_snps_aligned, ref_labels, n_refs_per_class, samples_list=ref_samples)
        self.reference_panel.reference_vcf = np.array(self.reference_panel.reference_vcf, dtype=np.int8)

        if genetic_map:
            map_df = read_map(genetic_map)
            self.reference_panel_pos = calculate_genetic_distances(map_df, self.mixed_pos)
        else:
            self.reference_panel_pos = self.mixed_pos

        self.mixed_labels = None # 在这个新流程中，labels对齐逻辑复杂，暂时禁用
        self.single_arc = single_arc
        self.transforms = transforms
        
        # --- 7. 准备最终的info对象 ---
        self.info = {
            'chm': [ref_info['chm'][0]] if isinstance(ref_info['chm'], (list, np.ndarray)) else [ref_info['chm']],
            'pos': list(common_positions),
            'samples': self.samples_to_load # 只包含当前块的样本名
        }
        logging.info(f"Dataset chunk for samples {self.samples_to_load} is ready.")

    def __len__(self):
        return self.mixed_vcf.shape[0]

    def __getitem__(self, index):

        item = {
            "mixed_vcf": self.mixed_vcf[index].astype(float),
        }
        # 如果需要处理标签，确保self.mixed_labels也已正确对齐
        if self.mixed_labels is not None:
            item["mixed_labels"] = self.mixed_labels[index]

        # 从已对齐的参考面板中抽样
        item["ref_panel"], item['reference_names'], item[
            'reference_idx'] = self.reference_panel.sample_reference_panel()

        # 使用对齐后的位置信息
        item["pos"] = self.reference_panel_pos[:]
        item["single_arc"] = self.single_arc

        item = ref_pan_to_tensor(item)

        if self.transforms is not None:
            item = self.transforms(item)
            
        return item

    """
    def __getitem__(self, index):
        # 获取第 index 个样本的数据
        mixed_vcf_sample = self.mixed_vcf[index]
        if self.mixed_labels is not None:
            mixed_labels_sample = self.mixed_labels[index]
        else:
            mixed_labels_sample = None

        # 执行针对该样本的过滤操作
        filtered_vcf, filtered_labels, filtered_pos, filtered_ref_panel = self.filter_sample(
            mixed_vcf_sample, mixed_labels_sample, self.mixed_pos, index
        )

        # 构建返回的字典
        item = {
            "mixed_vcf": filtered_vcf.astype(float),
            "pos": filtered_pos,
            "ref_panel": filtered_ref_panel,
            "single_arc": self.single_arc
        }
        if filtered_labels is not None:
            item["mixed_labels"] = filtered_labels

        # 转换为张量
        item = ref_pan_to_tensor(item)

        # 应用转换（如果有）
        if self.transforms is not None:
            item = self.transforms(item)
        return item
    """

    def _get_sample_cache_file(self, mixed_file_path, index):
        """为每个样本生成唯一的缓存文件名"""
        file_hash = hashlib.md5(mixed_file_path.encode()).hexdigest()
        cache_file = os.path.join(self.reference_panel.cache_dir, f"{file_hash}_sample_{index}_filtered_cache.pkl")
        return cache_file

    def filter_sample(self, mixed_vcf_sample, mixed_labels_sample, mixed_pos, index):
        # 检查是否存在缓存
        cache_file = self._get_sample_cache_file(self.mixed_file_path, index)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    return cached_data['vcf'], cached_data['labels'], cached_data['pos'], cached_data['ref_panel'], cached_data['info']
            except (EOFError, pickle.UnpicklingError):
                print(f"缓存文件损坏，重新生成: {cache_file}")
                os.remove(cache_file)
        else:
            print(f"未找到缓存，执行过滤操作: {cache_file}")

        # 执行针对单个样本的过滤操作
        filtered_vcf, filtered_labels, filtered_pos, filtered_ref_panel, filtered_info = self.filter_reference_panel(
            mixed_vcf_sample, mixed_labels_sample, mixed_pos, self.mixed_file_path, index
        )
        print("Filter done.")
        # 保存到缓存
        with open(cache_file, "wb") as f:
            pickle.dump({
                'vcf': filtered_vcf,
                'labels': filtered_labels,
                'pos': filtered_pos,
                'ref_panel': filtered_ref_panel,
                'info' : filtered_info
            }, f)
        
        return filtered_vcf, filtered_labels, filtered_pos, filtered_ref_panel, filtered_info


    def save_filtered_positions(self, filtered_vcf, filtered_labels, filtered_pos, filtered_ref_panel, mixed_file_path, index):
        """
        将过滤后的位点信息保存为 CSV 格式，文件名包含哈希和索引，便于后续查看和分析。
        """
        # 为每个样本生成唯一的文件名
        file_hash = hashlib.md5(mixed_file_path.encode()).hexdigest()
        file_name = f"{file_hash}_sample_{index}_dropped_positions.csv"
        file_path = os.path.join(self.reference_panel.cache_dir, file_name)

        # 打印参数 shape 和示例值的日志
        print("Logging parameter shapes and examples:")
        print(f"filtered_vcf shape: {filtered_vcf.shape}")
        print(f"filtered_labels shape: {filtered_labels.shape}")
        print(f"filtered_pos shape: {filtered_pos.shape}")
        # print(f"filtered_ref_panel shape: {filtered_ref_panel.shape}")

        # 准备保存数据的 DataFrame
        df = pd.DataFrame({
            'index': range(len(filtered_vcf)),
            'position': filtered_pos,
            'vcf': filtered_vcf,  # 处理为逗号分隔的字符串
            'label': filtered_labels if filtered_labels is not None else [''] * len(filtered_vcf),
        })

        # 输出 DataFrame 的一些信息进行调试
        
        print(f"afr:{filtered_ref_panel[0]}")
        print(f"den:{filtered_ref_panel[1]}")
        print(f"nean:{filtered_ref_panel[2]}")
        # 对每个参考面板数据（afr, den, nean）取唯一值并处理
        ref_afr_data = self.process_reference_data(filtered_ref_panel[0])  # AFR 数据
        ref_den_data = self.process_reference_data(filtered_ref_panel[1])  # DEN 数据
        ref_nean_data = self.process_reference_data(filtered_ref_panel[2])  # NEAN 数据

        # 将处理后的参考数据添加到 DataFrame
        df['ref_afr'] = ref_afr_data
        df['ref_den'] = ref_den_data
        df['ref_nean'] = ref_nean_data

        # 输出更新后的 DataFrame 信息
        print(f"Updated DataFrame columns: {df.columns}")
        print(f"Updated DataFrame shape: {df.shape}")

        # 保存到 CSV 文件
        df.to_csv(file_path, index=False)

        print(f"已将过滤后的位点信息保存为 CSV 文件：{file_path}")

    def process_reference_data(self, ref_data):
        """
        处理参考数据，将相同的基因型数据合并为一个逗号分隔的字符串。
        """
        ref_data_processed = []

        # 对每一行的参考数据进行处理
        for row in ref_data.T:  # 转置，假设每列是一个样本
            unique_values = np.unique(row)  # 获取唯一值
            ref_data_processed.append(",".join(map(str, unique_values)))  # 合并成逗号分隔的字符串

        return ref_data_processed


    def filter_reference_panel(self, mixed_vcf_sample, mixed_labels_sample, mixed_pos, mixed_file_path, index):
        # 将 mixed_vcf_sample 转换为 GPU 上的张量
        mixed_vcf_tensor = torch.tensor(mixed_vcf_sample, dtype=torch.float32, device='cuda')

        # 获取参考样本索引
        afr_indices = self.reference_panel.reference_panel_index_dict.get(0, [])
        den_indices = self.reference_panel.reference_panel_index_dict.get(1, [])
        nean_indices = self.reference_panel.reference_panel_index_dict.get(2, [])

        # 内存溢出 修改掉
        # ref_vcf_tensor = self.reference_panel.reference_vcf_tensor

        # 提取参考样本数据（已在 GPU 上） 内存溢出
        # afr_refs = ref_vcf_tensor[afr_indices]    # 形状：(afr_count, num_sites)
        # den_refs = ref_vcf_tensor[den_indices]    # 形状：(den_count, num_sites)
        # nean_refs = ref_vcf_tensor[nean_indices]  # 形状：(nean_count, num_sites)
        afr_refs_cpu = self.reference_panel.reference_vcf[afr_indices]    # 形状：(afr_count, num_sites)
        den_refs_cpu = self.reference_panel.reference_vcf[den_indices]    # 形状：(den_count, num_sites)
        nean_refs_cpu = self.reference_panel.reference_vcf[nean_indices]  # 形状：(nean_count, num_sites)

         # 将部分数据转移到 GPU
        afr_refs = torch.tensor(afr_refs_cpu, dtype=torch.float32, device='cuda')    # 形状：(afr_count, num_sites)
        den_refs = torch.tensor(den_refs_cpu, dtype=torch.float32, device='cuda')    # 形状：(den_count, num_sites)
        nean_refs = torch.tensor(nean_refs_cpu, dtype=torch.float32, device='cuda')

        # 初始化掩码为全 True，直接在 GPU 上
        num_sites = mixed_vcf_tensor.shape[0]
        mask = torch.ones(num_sites, dtype=torch.bool, device='cuda')

        # 基因型编码为 0 和 1
        genotypes = torch.tensor([0, 1], dtype=torch.float32, device='cuda')  # 形状：(2,)

        # 定义块大小
        chunk_size = 100000  # 根据实际情况调整
        num_chunks = (num_sites + chunk_size - 1) // chunk_size  # 计算需要的块数

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_sites)
            chunk_slice = slice(start_idx, end_idx)

            # 提取当前块的数据
            afr_refs_chunk = afr_refs[:, chunk_slice]          # 形状：(afr_count, chunk_size)
            den_refs_chunk = den_refs[:, chunk_slice]          # 形状：(den_count, chunk_size)
            nean_refs_chunk = nean_refs[:, chunk_slice]        # 形状：(nean_count, chunk_size)
            mixed_vcf_chunk = mixed_vcf_tensor[chunk_slice]    # 形状：(chunk_size,)

            # 计算各族群的基因型存在矩阵
            # 使用广播机制进行向量化操作
            afr_presence = (afr_refs_chunk.unsqueeze(0) == genotypes[:, None, None]).any(dim=1)  # 形状：(2, chunk_size)
            den_presence = (den_refs_chunk.unsqueeze(0) == genotypes[:, None, None]).any(dim=1)  # 形状：(2, chunk_size)
            nean_presence = (nean_refs_chunk.unsqueeze(0) == genotypes[:, None, None]).any(dim=1)  # 形状：(2, chunk_size)

            # 计算共享基因型存在矩阵
            shared_presence = afr_presence & den_presence & nean_presence  # 形状：(2, chunk_size)

            # 待测样本的基因型
            mixed_genotypes = mixed_vcf_chunk.long()

            to_filter_mask = shared_presence[mixed_genotypes, torch.arange(mixed_genotypes.size(0))]  # (chunk_size,)
            global_indices = start_idx + torch.nonzero(to_filter_mask, as_tuple=False).squeeze(1)
            mask[global_indices] = False

            # 清理显存
            del afr_refs_chunk, den_refs_chunk, nean_refs_chunk, mixed_vcf_chunk
            torch.cuda.empty_cache()

        # 应用掩码
        filtered_info = {}
        # 不可以修改dataset层面的原始info 要保留原始的info
        # if self.info is not None:
        #     for key in info:
        #         filtered_info[key] = info[key][mask]
        #     self.info = filtered_info

        filtered_vcf = mixed_vcf_tensor[mask].cpu().numpy()
        if mixed_labels_sample is not None:
            mixed_labels_tensor = torch.tensor(mixed_labels_sample, dtype=torch.float32, device='cuda')
            filtered_labels = mixed_labels_tensor[mask].cpu().numpy()
        else:
            filtered_labels = None

        filtered_pos = mixed_pos[mask.cpu().numpy()]
        filtered_afr_refs = afr_refs[:, mask].cpu().numpy()
        filtered_den_refs = den_refs[:, mask].cpu().numpy()
        filtered_nean_refs = nean_refs[:, mask].cpu().numpy()
        #filtered_ref_panel_tensor = ref_vcf_tensor[:, mask].cpu().numpy()

        # 重构 ref_panel 字典，保持与原始格式一致
        filtered_ref_panel = {}
        sorted_ref_panel_indices = [0, 1, 2]
        filtered_ref_panel = {
            0: filtered_afr_refs,
            1: filtered_den_refs,
            2: filtered_nean_refs
        }
        filtered_ref_panel_dict = filtered_ref_panel
        filtered_ref_panel = np.vstack([filtered_ref_panel[i] for i in sorted_ref_panel_indices])

        print(f"过滤前待测样本SNP数：{mixed_vcf_sample.shape[0]} ;过滤后待测样本SNP数：{filtered_vcf.shape[0]} ;过滤后参考SNP数：{filtered_den_refs.shape[1]}")
        # 调用保存过滤后的位点信息方法，保存为 CSV
        # 反转 mask，表示将过滤掉的位点保存下来
        inverse_mask = ~mask  # 反转掩码
        # 提取被过滤掉的位点
        dropped_vcf = mixed_vcf_tensor[inverse_mask].cpu().numpy()
        if mixed_labels_sample is not None:
            mixed_labels_tensor = torch.tensor(mixed_labels_sample, dtype=torch.float32, device='cuda')
            dropped_labels = mixed_labels_tensor[inverse_mask].cpu().numpy()
        else:
            dropped_labels = None
        dropped_pos = mixed_pos[inverse_mask.cpu().numpy()]
        dropped_afr_refs = afr_refs[:, inverse_mask].cpu().numpy()
        dropped_den_refs = den_refs[:, inverse_mask].cpu().numpy()
        dropped_nean_refs = nean_refs[:, inverse_mask].cpu().numpy()

        # 重构被过滤掉的 ref_panel 字典
        dropped_ref_panel = {
            0: dropped_afr_refs,
            1: dropped_den_refs,
            2: dropped_nean_refs
        }
        dropped_ref_panel_dict = dropped_ref_panel
        dropped_ref_panel = np.vstack([dropped_ref_panel[i] for i in [0, 1, 2]])

        self.save_filtered_positions(dropped_vcf, dropped_labels, dropped_pos, dropped_ref_panel_dict, mixed_file_path, index)
        
        # 释放不再需要的显存
        del mixed_vcf_tensor, afr_refs, den_refs, nean_refs
        torch.cuda.empty_cache()
        return filtered_vcf, filtered_labels, filtered_pos, filtered_ref_panel, filtered_info

class ReferencePanelDatasetSmall(ReferencePanelDataset):
    def __init__(self, mixed_file_path, reference_panel_h5,
                 reference_panel_vcf, reference_panel_map,
                 n_refs_per_class, transforms, single_arc=0, genetic_map=None, labels=None, n_samples=16):

        # 调用父类构造函数
        super().__init__(mixed_file_path, reference_panel_h5, reference_panel_vcf, 
                         reference_panel_map, n_refs_per_class, transforms, 
                         single_arc, genetic_map, labels)

        # 只加载前 n_samples 个样本
        if n_samples is not None and n_samples > 0:
            self._limit_samples(n_samples)

    def _limit_samples(self, n_samples):
        # 限制 mixed_vcf 到前 n_samples 个
        self.mixed_vcf = self.mixed_vcf[:n_samples]

        if self.mixed_labels is not None:
            self.mixed_labels = self.mixed_labels[:n_samples]

        # 更新数据集的长度
        self.indices = list(range(len(self.mixed_vcf)))

    def __len__(self):
        # 返回限制后的样本数
        return len(self.mixed_vcf)

def reference_panel_collate(batch):
    ref_panel = []
    reference_names = []
    reference_idx = []
    for x in batch:
        ref_panel.append(x["ref_panel"])
        reference_names.append(x["reference_names"])
        reference_idx.append(x['reference_idx'])
        del x["ref_panel"]
        del x["reference_names"]
        del x['reference_idx']

    batch = default_collate(batch)
    batch["ref_panel"] = ref_panel
    batch["reference_names"] = reference_names
    batch["reference_idx"] = reference_idx

    return batch
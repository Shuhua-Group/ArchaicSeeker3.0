import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np

import math

import time

from .transformer import GeneralModelTransformer, GeneralModelMamba
from ..stepsagnostic import filter_loci
from mamba_ssm import Mamba
import os

class ResidualConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dilation=1):
        super(ResidualConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                              padding=0, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()
        if in_planes != out_planes:
            self.shortcut = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        total_padding = effective_kernel - 1
        left_pad = total_padding // 2
        right_pad = total_padding - left_pad
        x_padded = f.pad(x, (left_pad, right_pad))
        
        out = self.conv(x_padded)
        out = self.bn(out)
        out = self.relu(out)
        residual = self.shortcut(x)
        if out.size(-1) != residual.size(-1):
            diff = out.size(-1) - residual.size(-1)
            if diff > 0:
                out = out[..., : -diff]
            else:
                out = f.pad(out, (0, -diff))
        return out + residual

class ImprovedAncestryLevelSmoother(nn.Module):
    def __init__(self, in_planes=3, planes=32, kernel_sizes=[500, 1000, 1500]):
        super(ImprovedAncestryLevelSmoother, self).__init__()
        print("[INFO]: ImprovedAncestryLevelSmoother applied.")
        
        self.kernel_sizes = kernel_sizes
        self.in_planes = in_planes 
        self.planes = planes
        conv_input_channels = in_planes + 1

        self.branches = nn.ModuleList([
            nn.Sequential(
                ResidualConvBlock(conv_input_channels, planes, ks),
                ResidualConvBlock(planes, planes, ks)
            ) for ks in kernel_sizes
        ])

        self.fusion = nn.Conv1d(planes * len(kernel_sizes), planes, kernel_size=1)
        self.deconv = nn.ConvTranspose1d(planes, 3, kernel_size=1000, padding=0, bias=False)
        self.bn_out = nn.BatchNorm1d(3)

    def forward(self, x, pos=None, test=False):
        b, c, l = x.size()
        if pos is None:
            raise ValueError("Position tensor must be provided.")
        pos = pos.view(b, 1, l)
        x_cat = torch.cat([x, pos], dim=1)
        
        branch_outs = [branch(x_cat) for branch in self.branches] 
        multi_scale = torch.cat(branch_outs, dim=1)
        fused = self.fusion(multi_scale)
        deconv_out = self.deconv(fused)
        deconv_out = self.bn_out(deconv_out)

        if deconv_out.size(-1) > l:
            deconv_out = deconv_out[..., :l]
        elif deconv_out.size(-1) < l:
            deconv_out = f.pad(deconv_out, (0, l - deconv_out.size(-1)))
        
        out = x + deconv_out
        return out


class SlidingWindowSum(nn.Module):

    def __init__(self, win_size, stride):
        super(SlidingWindowSum, self).__init__()
        self.kernel = torch.ones(1, 1, win_size).float() / win_size
        # We pass is as parameter but freeze it
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        self.stride = stride

    def forward(self, inp):
        inp = inp.unsqueeze(1)
        inp = f.conv1d(inp, self.kernel, stride=(self.stride), padding=self.kernel.shape[-1]//2)
        inp = inp.squeeze(1)
        return inp

class EnCoder(nn.Module):
    def __init__(self, embed_dim=144, hidden_size=756, num_layer=12):
        super(EnCoder, self).__init__()

        self.pos_embed = nn.Parameter(torch.zeros((1, 158, embed_dim)))

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=144, nhead=12, dim_feedforward=hidden_size,
                                                            batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layer)

    def forward(self, b):
        b = b + self.pos_embed
        out = self.transformer_layer(b)
        return out


class SmootherBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, kernel_size=50, stride=1):
        super(SmootherBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(kernel_size,),
                               stride=(stride,), bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.ConvTranspose1d(planes, 1, kernel_size=(kernel_size,))
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        out = x + self.bn2(self.conv2(f.relu(self.bn1(self.conv1(x)))))
        return out


class AncestryLevelConvSmoother_(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=50, stride=1):
        super(AncestryLevelConvSmoother, self).__init__()
        self.in_planes = 1
        self.layer = self._make_layer(SmootherBlock, 1, 8, stride=1)
        self.layer_out = self._make_layer(SmootherBlock, 1, 1, stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = layer(x)
        out = layer_out(out)
        out = f.sigmoid(out)
        return out


class AncestryLevelConvSmoother(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=1000, stride=1):
        super(AncestryLevelConvSmoother, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(kernel_size,),
                               stride=(stride,), bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.ConvTranspose1d(planes, 3, kernel_size=(kernel_size,))
        self.bn2 = nn.BatchNorm1d(3)

    def forward(self, x, pos=None, test=False):
        b, l, s = x.size()
        POS = pos.reshape([b,1,s])
        out = torch.cat([x, POS], dim=1)
        out = self.bn2(self.conv2(f.relu(self.bn1(self.conv1(out)))))
        out = x + out
        return out


class RefMaxPool(nn.Module):
    def __init__(self):
        super(RefMaxPool, self).__init__()

    def forward(self, inp):
        maximums, indices = torch.max(inp, dim=0)
        return maximums.unsqueeze(0), indices


class BaggingMaxPool(nn.Module):
    def __init__(self, k=20, split=0.25):
        super(BaggingMaxPool, self).__init__()
        self.k = k
        self.split = split

        self.maxpool = RefMaxPool()
        self.averagepool = AvgPool()

    def forward(self, inp):
        pooled_refs = []

        total_n = inp.shape[0]
        select_n = int(total_n * self.split)

        for _ in range(self.k):
            indices = torch.randint(low=0, high=int(total_n), size=(select_n,))
            selected = inp[indices, :]
            maxpooled = self.maxpool(selected)

            pooled_refs.append(maxpooled)
        pooled_refs = torch.cat(pooled_refs, dim=0)
        return self.averagepool(pooled_refs)


class TopKPool(nn.Module):
    def __init__(self, k):
        super(TopKPool, self).__init__()
        self.k = k

    def forward(self, inp):
        k = self.k
        if inp.shape[0] < k:
            k = inp.shape[0]
        maximums, indices = torch.topk(inp, k=k, dim=0)
        assert indices.max() < inp.shape[0]
        return maximums, indices[0]


class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, inp):
        inp = (inp + 1)/2
        inp = inp.mean(dim=0, keepdim=True)
        return inp


def stack_ancestries(inp):
    """
    Stack ancestry predictions from reference panel comparison.

    FIX: Ensures output always has 3 channels (ancestries 0, 1, 2),
    even if reference panel is missing some ancestry groups.
    This handles single/dual archaic source scenarios.
    """
    out = []
    for i, x in enumerate(inp):
        # FIX: Always create 3 channels, even if some ancestries are missing
        # Ancestry indices: 0=African, 1=Denisovan, 2=Neanderthal
        out_sample = [None] * 3

        for ancestry in x.keys():
            if ancestry < 3:  # Safety check
                out_sample[ancestry] = x[ancestry]

        # Fill missing ancestries with zeros
        first_valid = None
        for idx, tensor in enumerate(out_sample):
            if tensor is not None:
                first_valid = tensor
                break

        if first_valid is None:
            raise ValueError("No valid ancestry data found in reference panel")

        # Replace None with zero tensors of the same shape
        for idx in range(3):
            if out_sample[idx] is None:
                out_sample[idx] = torch.zeros_like(first_valid)

        out_sample = torch.cat(out_sample)
        out.append(out_sample)
    out = torch.stack(out)

    return out

class Freq(nn.Module):
    def __init__(self):
        super(Freq, self).__init__()
    
    def forward(self, inp):
        inp = (inp + 1)/2
        inp = inp.mean(dim=0, keepdim=True)
        maximums, indices = torch.max(inp, dim=0)
        return inp, indices


class AddPoolings(nn.Module):
    def __init__(self, max_n=2):
        self.max_n = max_n
        super(AddPoolings, self).__init__()
        self.weights = nn.Parameter(torch.rand(max_n).unsqueeze(1), requires_grad=True)

    def forward(self, inp):
        out = inp * self.weights[:min(inp.shape[0], self.max_n)]
        out = torch.sum(out, dim=0, keepdim=True)

        return out


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.window_size = args.win_size

        self.inpref_oper = XOR()

        if args.ref_pooling == "maxpool":
            self.ref_pooling = RefMaxPool()
        elif args.ref_pooling == "topk":
            self.ref_pooling = TopKPool(args.topk_k)
        elif args.ref_pooling == "average":
            self.ref_pooling = AvgPool()
        elif args.ref_pooling == "freq":
            self.ref_pooling = Freq()
        elif args.ref_pooling == "topk-average":
            self.ref_pooling = TopKPool(args.topk_k)
            self.ref_pooling_avg = AvgPool()
        else:
            raise ValueError('Wrong type of ref pooling')

    def forward(self, input_mixed, ref_panel):

        with torch.no_grad():
            out = self.inpref_oper(input_mixed, ref_panel)
            out_ = []
            n, l = out[0][1].shape
            max_indices_batch = []
            for i,x in enumerate(out):
                x_ = {}
                x_avg = {}
                max_indices_element = []
                for c in x.keys():
                    x_[c] =  x[c]
                    if self.args.ref_pooling == "topk-average":
                        x_avg[c] = self.ref_pooling_avg(x_[c])
                    x_[c], max_indices = self.ref_pooling(x_[c])
                    max_indices_element.append(max_indices)

                out_.append(x_)
                out_.append(x_avg)
                max_indices_element = torch.stack(max_indices_element, dim=0)
                max_indices_batch.append(max_indices_element)

            max_indices_batch = torch.stack(max_indices_batch, dim=0)

        return out_, max_indices_batch


class AgnosticModel(nn.Module):

    def __init__(self, args):
        super(AgnosticModel, self).__init__()
        if args.win_stride == -1:
            args.win_stride = args.win_size
        self.args = args

        self.base_model = BaseModel(args=args)

        if args.dropout > 0:
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            self.dropout = nn.Sequential()
        
        self.unet = GeneralModelMamba(
                                        input_dim=7,
                                        output_dim=3,
                                        model_dim=192,          
                                        num_layers=8,
                                        conv_kernel=4,       
                                    )

    def forward(self, batch, test=False, infer=False):
        
        with torch.no_grad():
            input_mixed, ref_panel, pos, single_arc, labels = batch["mixed_vcf"], batch["ref_panel"], batch["pos"], batch["single_arc"], batch['mixed_labels']

            if not infer:
                input_mixed, ref_panel = add_noise_with_probability(input_mixed, ref_panel, noise_probability=0.5, max_flip_rate=0.02)
            pos = pos[0].float()
        
            single_arc = single_arc[0]

            out_avg = []
            
            out, max_indices = self.base_model(input_mixed, ref_panel)

            out_avg_list = [] 
            out_list = []

            for i in range(0, len(out), 2):
                group = out[i:i+2] 
                if len(group[1]) != 0:
                    out_avg_list.append(stack_ancestries([group[1]]))
                out_list.append(stack_ancestries([group[0]]))

            out = torch.cat(out_list, dim=0)
            out = (out + 1)/2

            Den = 1
            Nean = 2
            if single_arc==2:
                out[:,Nean,:] = out[:,Den,:]
            if single_arc==1:
                out[:,Den,:] = out[:,Nean,:]
            if len(out_avg_list)>0:
                out_avg = torch.cat(out_avg_list, dim=0)
                if single_arc==2:
                    out_avg[:,Nean,:] = out_avg[:,Den,:]
                if single_arc==1:
                    out_avg[:,Den,:] = out_avg[:,Nean,:]
                out = torch.cat([out, out_avg],dim=1)

            window_size = self.args.win_size
            win_stride = self.args.win_stride
            out = self.dropout(out)
            
            POS = (pos / 1000000 / 100).to(out.device) + torch.zeros([out.shape[0],1,out.shape[2]], dtype=torch.float32).to(out.device)

            pad_num0 = 0
            if not infer:
                out, POS, pad_num0 = random_pad(out, POS, max_pad=window_size)

            pad_num = win_stride - out.size()[2]%win_stride
            out = f.pad(out, (0, pad_num), 'constant', 1)
            POS = f.pad(POS, (0, pad_num), 'replicate')


            out = torch.cat([out, POS], dim=1)

            out = out.unfold(2, window_size, self.args.win_stride)
            out = out.permute(0, 2, 1, 3)
            B = out.shape[0]

            out = out.reshape(-1, 7, window_size)

            slice_to_scale = out[:, -1, :]
            min_vals = torch.min(slice_to_scale, dim=1).values
            min_vals = min_vals.view(-1, 1) # [B,1]
            scaled_slice = slice_to_scale - min_vals 
            out[:, -1, :] = scaled_slice
            
        inputs = out.cuda() 
        out = process_in_batches(inputs, self.unet, 100) 
        n_windows, C, W = out.shape
        out = out.reshape(B, n_windows, C, window_size)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, C, n_windows * W)
        out_basemodel = out_flat[:,:,pad_num0:-pad_num]
        output = {
            'predictions': out_basemodel,
            'pad_num': pad_num,
            'out_basemodel': out_basemodel,
            #'out_smoother': out_smoother,
            'max_indices': max_indices,
            'labels':labels
        }

        return output

def ensure_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = ensure_device(v, device)
        return obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = ensure_device(v, device)
        return obj
    else:
        return obj

class AgnosticModelInferUsed(nn.Module):
    def __init__(self, args):
        super(AgnosticModelInferUsed, self).__init__()
        if args.win_stride == -1:
            args.win_stride = args.win_size
        self.args = args

        # 1) Basemodel
        self.base_model = BaseModel(args=args)

        # 2) Dropout
        if args.dropout > 0:
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            self.dropout = nn.Sequential()

        # 3) 核心UNet/Transformer
        self.unet = GeneralModelTransformer(
            input_dim=7, 
            output_dim=3, 
            max_seq_len=args.win_size
        )

    def forward(self, batch, test=False, infer=True):

        device = next(self.parameters()).device 
        batch = ensure_device(batch, device)

        input_mixed = batch["mixed_vcf"]   # (B,L)
        ref_panel   = batch["ref_panel"]   # (dict or tensor)
        pos         = batch["pos"]         # (B,L)
        single_arc  = batch["single_arc"]  # (B,) or shape(1,)
        labels      = batch["mixed_labels"]# (B,L) or None

        with torch.no_grad():
            if not infer:
                input_mixed, ref_panel = add_noise_with_probability(
                    input_mixed, ref_panel, 
                    noise_probability=0.5, 
                    max_flip_rate=0.02
                )

            out, max_indices = self.base_model(input_mixed, ref_panel)

            out_avg_list = []
            out_list = []
            for i in range(0, len(out), 2):
                group = out[i:i+2]
                if len(group[1]) != 0:
                    out_avg_list.append(stack_ancestries([group[1]]))
                out_list.append(stack_ancestries([group[0]]))

            out = torch.cat(out_list, dim=0)
            out = (out + 1)/2

            single_arc_val = single_arc[0].item() if single_arc.dim()>0 else single_arc.item()
            Den = 1
            Nean= 2
            if single_arc_val == 2:
                out[:,Nean,:] = out[:,Den,:]
            if single_arc_val == 1:
                out[:,Den,:]  = out[:,Nean,:]

            if len(out_avg_list)>0:
                out_avg = torch.cat(out_avg_list, dim=0)
                if single_arc_val == 2:
                    out_avg[:,Nean,:] = out_avg[:,Den,:]
                if single_arc_val == 1:
                    out_avg[:,Den,:]  = out_avg[:,Nean,:]
                out = torch.cat([out, out_avg], dim=1)

            out = self.dropout(out)  
            pos_flt = pos[0].float()
            POS = (pos_flt / 1e8).to(device)
            POS = POS.unsqueeze(0).unsqueeze(0).expand(out.shape[0], 1, out.shape[2])
            pad_num0 = 0
            L_out = out.shape[2]
            window_size = self.args.win_size  
            win_stride  = self.args.win_stride 

            pad_num = win_stride - (L_out % win_stride) if (L_out % win_stride)!=0 else 0
            if pad_num>0:
                out  = f.pad(out, (0, pad_num), 'constant', 1)
                POS  = f.pad(POS, (0, pad_num), 'replicate')

            out = torch.cat([out, POS], dim=1)

            out = out.unfold(2, window_size, win_stride)
            out = out.permute(0, 2, 1, 3)
            B_ = out.shape[0]
            N_windows= out.shape[1]
            C_ = out.shape[2]
            out = out.reshape(-1, C_, window_size)  # (B_*N_windows, C_, window_size)

            out = out.to(device)
            out = process_in_batches(out, self.unet, 100)

            n_w, C_w, W_w = out.shape
            out = out.reshape(B_, N_windows, C_w, W_w)
            out_flat = out.permute(0, 2, 1, 3).reshape(B_, C_w, N_windows*W_w)

            total_length = out_flat.shape[-1]
            sum_pred   = torch.zeros(B_, C_w, total_length, device=device)
            count_pred = torch.zeros(B_, C_w, total_length, device=device)
            window_positions = (torch.arange(N_windows, device=device).unsqueeze(1)*win_stride) \
                              + torch.arange(W_w, device=device).unsqueeze(0)
            window_positions_flat = window_positions.flatten()
            index = window_positions_flat.unsqueeze(0).unsqueeze(0).expand(B_, C_w, -1)
            sum_pred.scatter_add_(2, index, out_flat)
            ones = torch.ones_like(out_flat)
            count_pred.scatter_add_(2, index, ones)

            mask = (count_pred>0)
            final_out = torch.zeros_like(sum_pred)
            final_out[mask] = sum_pred[mask]/count_pred[mask]
            if pad_num>0:
                final_out = final_out[...,:-pad_num]
            out_basemodel = final_out

        output = {
            "predictions": out_basemodel,
            "pad_num": pad_num,
            "out_basemodel": out_basemodel,
            "max_indices": max_indices,
            "labels": labels
        }
        return output



class AgnosticModelInferNoSmoother(nn.Module):
    def __init__(self, args):
        super(AgnosticModelInferNoSmoother, self).__init__()
        if args.win_stride == -1:
            args.win_stride = args.win_size
        self.args = args

        self.base_model = BaseModel(args=args)

        #self.dropout = nn.Sequential()
        if args.dropout>0:
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            self.dropout = nn.Sequential()

        self.unet = GeneralModelTransformer(
            input_dim=7,
            output_dim=3,
            max_seq_len=args.win_size
        )

        self.unet = GeneralModelMamba(
                                        input_dim=7,
                                        output_dim=3,
                                        model_dim=192,
                                        num_layers=8,
                                        conv_kernel=4,       
                                    )

    def forward(self, batch, test=False, infer=False):
        with torch.no_grad():
            input_mixed, ref_panel, pos, single_arc = (
                batch["mixed_vcf"], batch["ref_panel"], batch["pos"], batch["single_arc"]
            )
            pos = pos[0].float()  # shape(L,)
            single_arc = single_arc[0]

            out_avg_list = []
            out, max_indices = self.base_model(input_mixed, ref_panel)
            if len(out[1]) != 0:
                out_avg_list.append(stack_ancestries([out[1]]))
            out = stack_ancestries([out[0]])
            out = (out + 1)/2

            Den = 1
            Nean= 2
            if single_arc==2:
                out[:,Nean,:] = out[:,Den,:]
            if single_arc==1:
                out[:,Den,:] = out[:,Nean,:]

            # FIX: Handle single archaic source case - ensure out_avg always has 3 channels
            if len(out_avg_list)>0:
                out_avg = torch.cat(out_avg_list, dim=0)
                if single_arc==2:
                    out_avg[:,Nean,:] = out_avg[:,Den,:]
                if single_arc==1:
                    out_avg[:,Den,:] = out_avg[:,Nean,:]

                # Ensure out_avg has exactly 3 channels (for single archaic source scenarios)
                # If it has fewer channels, pad with the appropriate channels from out_avg
                if out_avg.shape[1] < 3:
                    # Create a full 3-channel tensor, initialized with zeros or copied channels
                    B_avg, C_avg, L_avg = out_avg.shape
                    out_avg_full = torch.zeros(B_avg, 3, L_avg, device=out_avg.device, dtype=out_avg.dtype)

                    # Copy available channels
                    out_avg_full[:, :C_avg, :] = out_avg

                    # For missing channels, copy from the existing ones (handles single archaic)
                    if C_avg == 1:
                        # If only 1 channel, replicate it to all 3
                        out_avg_full[:, 1:, :] = out_avg[:, 0:1, :].expand(-1, 2, -1)
                    elif C_avg == 2:
                        # If 2 channels, copy the last one
                        out_avg_full[:, 2, :] = out_avg[:, 1, :]

                    out_avg = out_avg_full

                out = torch.cat([out, out_avg], dim=1)

            out = self.dropout(out)
            window_size = self.args.win_size
            win_stride  = self.args.win_stride

            POS = (pos / 1000000 / 100).to(out.device)
            POS = POS.unsqueeze(0).unsqueeze(0).expand(out.shape[0], 1, out.shape[2])

            pad_num = win_stride - out.size()[2]%win_stride
            out = f.pad(out, (0, pad_num), 'constant', 1)
            POS = f.pad(POS, (0, pad_num), 'replicate')

            out = torch.cat([out, POS], dim=1)

            out = out.unfold(2, window_size, win_stride) 
            out = out.permute(0, 2, 1, 3)                
            B_ = out.shape[0]
            n_windows= out.shape[1]
            C_ = out.shape[2]
            out = out.reshape(-1, C_, window_size)

            slice_to_scale = out[:,-1,:]
            min_vals = slice_to_scale.min(dim=1).values.view(-1,1)
            out[:,-1,:] = slice_to_scale - min_vals

        inputs = out.cuda()
        out = process_in_batches(inputs, self.unet, 50)
        n_windows, C_w, W_w = out.shape
        out = out.reshape(B_, n_windows, C_w, W_w)

        window_positions = (torch.arange(n_windows, device=out.device).unsqueeze(1)*win_stride) \
                         + torch.arange(W_w, device=out.device).unsqueeze(0)
        window_positions_flat = window_positions.flatten()

        out_flat = out.permute(0,2,1,3).reshape(B_, C_w, n_windows*W_w)
        total_length = POS.shape[2]
        sum_pred = torch.zeros(B_, C_w, total_length, device=out.device)
        count_pred= torch.zeros(B_, C_w, total_length, device=out.device)

        index = window_positions_flat.unsqueeze(0).unsqueeze(0).expand(B_, C_w, -1)
        sum_pred.scatter_add_(2, index, out_flat)
        ones = torch.ones_like(out_flat)
        count_pred.scatter_add_(2, index, ones)

        mask = (count_pred>0)
        final_out = torch.zeros_like(sum_pred)
        final_out[mask] = sum_pred[mask]/count_pred[mask]

        del sum_pred, count_pred, window_positions, window_positions_flat, out_flat, index, ones, mask
        torch.cuda.empty_cache()
        if pad_num>0:
            final_out = final_out[...,:-pad_num]

        output = {
            "predictions": final_out,   
            "pad_num": pad_num,
            "out_basemodel": final_out, 
            "out_smoother": None,       
            "max_indices": max_indices,
            "labels": batch.get("mixed_labels", None)
        }
        return output

def process_in_batches(inputs, model, batch_size):
    """
    Process the input tensor in smaller batches.

    Args:
    inputs (torch.Tensor): Input tensor of shape (batch, 3, 500).
    model (torch.nn.Module): The model to process the inputs.
    batch_size (int): Size of each smaller batch.

    Returns:
    torch.Tensor: Output tensor after processing all batches.
    """
    # Calculate the number of batches
    total_batches = inputs.size(0) // batch_size + (1 if inputs.size(0) % batch_size != 0 else 0)

    outputs = []
    for i in range(total_batches):
        batch_inputs = inputs[i * batch_size:(i + 1) * batch_size]
        batch_outputs = model(batch_inputs)
        outputs.append(batch_outputs)

    return torch.cat(outputs, dim=0)


def multiply_ref_panel_stack_ancestries(mixed, ref_panel):
    all_refs = [None] * len(ref_panel.keys())
    for ancestry in ref_panel.keys():
        all_refs[ancestry] = ref_panel[ancestry]
    all_refs = torch.cat(all_refs, dim=0)

    return all_refs * mixed.unsqueeze(0)


def multiply_ref_panel(mixed, ref_panel):
    out = {
        ancestry: mixed.unsqueeze(0) * ref_panel[ancestry] for ancestry in ref_panel.keys()
    }
    return out


# SNP-wise similrity
class XOR(nn.Module):

    def __init__(self):
        super(XOR, self).__init__()

    def forward(self, input_mixed, ref_panel):
        with torch.no_grad():
            out = []
            for inp, ref in zip(input_mixed, ref_panel):
                multi = multiply_ref_panel(inp, ref)
                out.append(multi)
        return out

def interpolate_and_pad(inp, upsample_factor, target_len):
    bs, n_chann, original_len = inp.shape
    non_padded_upsampled_len = original_len * upsample_factor
    inp = f.interpolate(inp, size=non_padded_upsampled_len)

    left_pad = (target_len - non_padded_upsampled_len) // 2
    right_pad = target_len - non_padded_upsampled_len - left_pad
    inp = f.pad(inp, (left_pad, right_pad), mode="replicate")

    return inp

def change_the_features(old_data):
    first_row = old_data[:, 0, :].unsqueeze(1)
    
    other_rows = old_data[:, 1:, :]
    
    condition_1 = (first_row == 0) & (other_rows == 1)
    condition_2 = (first_row == 1) & (other_rows == 0)
    condition_3 = (first_row == 0) & (other_rows == 0)
    condition_4 = (first_row == 1) & (other_rows == 1)
    
    new_data = torch.zeros_like(other_rows).to(old_data.device)
    new_data[condition_1] = 0
    new_data[condition_2] = 1/3
    new_data[condition_3] = 2/3
    new_data[condition_4] = 1
    
    return new_data

def add_noise_with_probability(input_mixed, ref_panel, noise_probability=0.8, max_flip_rate=0.02):

    def add_noise_to_tensor(tensor, flip_rate):
        if flip_rate <= 0:
            return tensor.clone()
        noisy_tensor = tensor.clone()
        mask = torch.rand_like(noisy_tensor, dtype=torch.float) < flip_rate
        noisy_tensor[mask] *= -1
        
        return noisy_tensor
    add_noise_to_input = torch.rand(1).item() < noise_probability
    flip_rate_input = (torch.rand(1).item() * max_flip_rate) if add_noise_to_input else 0.0
    noisy_input_mixed = add_noise_to_tensor(input_mixed, flip_rate_input)
    noisy_ref_panel = []
    for panel_dict in ref_panel:
        noisy_panel_dict = {}
        for key, tensor in panel_dict.items():
            add_noise_to_ref = torch.rand(1).item() < noise_probability
            flip_rate_ref = (torch.rand(1).item() * max_flip_rate) if add_noise_to_ref else 0.0
            noisy_panel_dict[key] = add_noise_to_tensor(tensor, flip_rate_ref)
        noisy_ref_panel.append(noisy_panel_dict)
    
    return noisy_input_mixed, noisy_ref_panel


def random_pad(out, POS, max_pad=250):
    pad_len = torch.randint(0, max_pad + 1, (1,)).item()  
    padding = (pad_len, 0)
    padded_out = f.pad(out, padding, mode='constant', value=1)
    padded_POS = f.pad(POS, padding, mode='replicate')
    return padded_out, padded_POS, pad_len
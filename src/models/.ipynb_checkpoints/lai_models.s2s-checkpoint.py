import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np

import math

import time

from .resUnet1d import ResUnet,UNet
import torch.optim as optim

import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim)

    def forward(self, x):
        outputs, hidden = self.gru(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(output_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        prediction = self.out(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input, decoder_target, teacher_forcing_ratio=1.0):
        # Forward pass through the encoder
        # encoder_hidden, encoder_cell = self.encoder(encoder_input)
        encoder_hidden = self.encoder(encoder_input)
        # Initialize outputs tensor
        outputs = torch.zeros(decoder_input.shape[0], decoder_input.shape[1], self.decoder.output_dim).to(decoder_input.device)

        decoder_input_step = decoder_input[0, :]

        for t in range(1, decoder_input.shape[0]):
            # decoder_output, encoder_hidden, encoder_cell = self.decoder(decoder_input_step.unsqueeze(0), encoder_hidden, encoder_cell)
            decoder_output, encoder_hidden = self.decoder(decoder_input_step.unsqueeze(0), encoder_hidden)
            outputs[t] = decoder_output.squeeze(0)

            # Forward batch of sequences through decoder one time step at a time
            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

            if use_teacher_forcing and decoder_target is not None:
                decoder_input_step = decoder_target[t, :]
            else:
                decoder_input_step = decoder_output.squeeze(0)

        return outputs

class Seq2SeqCriterion(nn.Module):
    def __init__(self):
        super(Seq2SeqCriterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = 0
        for t in range(targets.shape[0]):
            decoder_output = outputs[t] # Remove the time step dimension
            target = targets[t].argmax(dim=1)  # Assuming target data is one-hot encoded
            loss += self.criterion(decoder_output, target)
        return loss


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
    def __init__(self, in_planes, planes, kernel_size=100, stride=1):
        super(AncestryLevelConvSmoother, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(kernel_size,),
                               stride=(stride,), bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.ConvTranspose1d(planes, 2, kernel_size=(kernel_size,))
        self.bn2 = nn.BatchNorm1d(2)
        # self.maxpool = nn.MaxPool1d(21, stride=21, padding=0)

    def forward(self, x, pos):
        b, l, s = x.size()
        POS = f.pad(pos, (0, s - pos.size()[2]%s), 'constant', 0)
        POS = POS.reshape([b,1,s])
        out = torch.cat([x, POS], dim=1)
        # out = x
        out = self.bn2(self.conv2(f.relu(self.bn1(self.conv1(out)))))
        # uot = self.maxpool(out)
        # out = x + out
        return out


class RefMaxPool(nn.Module):
    def __init__(self):
        super(RefMaxPool, self).__init__()

    def forward(self, inp):
        maximums, indices = torch.max(inp, dim=0)
        # print(indices)
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
        inp = inp.mean(dim=0, keepdim=True)
        return inp


def stack_ancestries(inp):
    out = []

    # inp is a batch (16)
    for i, x in enumerate(inp):
        out_sample = [None] * len(x.keys())
        for ancestry in x.keys():
            out_sample[ancestry] = x[ancestry]
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
        # self.weights=nn.Parameter(torch.ones(max_n).unsqueeze(1))
        self.weights = nn.Parameter(torch.rand(max_n).unsqueeze(1), requires_grad=True)
        # self.bias = nn.Parameter(torch.rand(max_n).unsqueeze(1), requires_grad=True)

    def forward(self, inp):
        # inp = inp + self.bias[:min(inp.shape[0], self.max_n)]
        out = inp * self.weights[:min(inp.shape[0], self.max_n)]
        out = torch.sum(out, dim=0, keepdim=True)

        return out


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.window_size = args.win_size
        self.sliding_window_sum = SlidingWindowSum(win_size=21, stride=1)

        self.inpref_oper = XOR()

        if args.ref_pooling == "maxpool":
            self.ref_pooling = RefMaxPool()
        elif args.ref_pooling == "topk":
            self.ref_pooling = TopKPool(args.topk_k)
            self.add_poolings = AddPoolings(max_n=args.topk_k)
        elif args.ref_pooling == "average":
            self.ref_pooling = AvgPool()
        elif args.ref_pooling == "freq":
            self.ref_pooling = Freq()

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
            max_indices_element = []
            for c in x.keys():
                x_[c] = x[c] # self.sliding_window_sum(x[c]) # 
                x_[c], max_indices = self.ref_pooling(x_[c])
                # new_ref = torch.zeros([1, l]).to(input_mixed.device)
                # for index,item in enumerate(max_indices):
                #     new_ref[0, self.window_size * index: self.window_size * (index + 1)] = ref_panel[i][c][item][self.window_size * index: self.window_size * (index + 1)]
                # x_[c] = new_ref
                # if self.args.ref_pooling == 'topk':
                #     x_[c] = self.add_poolings(x_[c])
                max_indices_element.append(max_indices)

            out_.append(x_)
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
            print("Dropout=", args.dropout)
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            print("No dropout")
            self.dropout = nn.Sequential()

        self.encoder = Encoder(3, 128)
        self.decoder = Decoder(2, 128)
        self.unet = Seq2Seq(self.encoder, self.decoder)

        if args.smoother == "anc1conv":
            self.smoother = AncestryLevelConvSmoother(3, 32)
        elif args.smoother == "none":
            self.smoother = nn.Sequential()
        else:
            raise ValueError()

    def forward(self, batch, infer=False):
        

        input_mixed, ref_panel, pos = batch["mixed_vcf"], batch["ref_panel"], batch["pos"]

        pos = pos[0]

        # a=time.time()
        out, max_indices = self.base_model(input_mixed, ref_panel)
        # print(time.time()-a)

        out = stack_ancestries(out) # .to(next(self.parameters()).device)
        out = self.dropout(out)

        POS = (pos / 1000000 / 100).to(out.device) + torch.zeros([out.shape[0],1,out.shape[2]], dtype=torch.float32).to(out.device)
        out = torch.cat([out, POS], dim=1)
        
        window_size = self.args.win_size
        out = f.pad(out, (0, window_size - out.size()[2]%window_size), 'constant', 0)

        out = out.unfold(2, window_size, window_size)

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, 3, window_size).permute(2,0,1)

        decoder_input_data = torch.zeros(out.shape[0], out.shape[1], 2).to(out.device)
        num_elements = torch.numel(decoder_input_data)
        num_ones = int(num_elements * 0.02)
        indices = torch.randint(0, num_elements, (num_ones,))
        indices_seq = indices // (out.shape[1] * 2)
        indices_batch  = (indices % (out.shape[1] * 2))//2
        indices_last_dim = indices % 2
        # indices_3d = torch.stack((indices // (out.shape[1] * 2), (indices % (out.shape[1] * 2))//2, indices % 2))
        decoder_input_data[indices_seq, indices_batch, indices_last_dim] = 1

        out = self.unet(out, decoder_input_data, None, 0.5)

        out = out.permute(1,2,0)
        out_basemodel = out # (batch, win_size)
        seq_len = out.shape[-1]

        out = self.smoother(out, POS)
        # out = interpolate_and_pad(out, 5, seq_len)

        # out_basemodel = out # (batch, win_size)
        out_smoother = out

        if not infer:
            output = {
                'predictions': out,
                'out_basemodel': out_basemodel,
                'out_smoother': out_smoother,
                'max_indices': max_indices
            }

            return output
        else:
            return out


def multiply_ref_panel_stack_ancestries(mixed, ref_panel):
    all_refs = [None] * len(ref_panel.keys())
    for ancestry in ref_panel.keys():
        all_refs[ancestry] = ref_panel[ancestry]
    # all_refs = [ref_panel[ancestry] for ancestry in ref_panel.keys()]
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

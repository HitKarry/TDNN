import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import cupy_module.adacof_highdof as adacof
import cupy_module.adacof_classical as adacof_classical
from utility import CharbonnierFunc, moduleNormalize
import time


def make_model(args):
    return AdaCoFNet(args)

class input_embedding(nn.Module):
    def __init__(self, input_dim, d_model, max_len, device):
        super().__init__()
        self.device = device
        channel_pos = torch.arange(max_len)[None, :, None]
        self.register_buffer('channel_pos', channel_pos)
        self.emb_channel = nn.Embedding(max_len, d_model)
        self.linear = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.linear(x) + self.emb_channel(self.channel_pos)
        return self.norm(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (N, C, S, D)
        Returns:
            (N, C, S, D)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)
        # (N, h, C, S, d_k)
        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k).permute(0, 3, 1, 2, 4)
             for l, x in zip(self.linears, (query, key, value))]

        # (N, h, C, S, d_k)
        x = self.attn(query, key, value, dropout=self.dropout)

        # (N, S, T, D)
        x = x.permute(0, 2, 3, 1, 4).contiguous() \
             .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        return self.linears[-1](x)

def C_Attention(query, key, value, dropout=None):
    # (N, h, C, S, d_k)
    N, h, C, S, d_k = query.shape
    query = query.transpose(2,3)
    key = key.transpose(2,3)
    value = value.transpose(2,3)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(2,3)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
class Channel_Attention_Layer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.c_attn = MultiHeadedAttention(d_model, nheads, C_Attention, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.c_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class Attention(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=False),
            )

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.Softmax(dim=1),
            )

        def Subnet_occlusion():
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.Sigmoid(),
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleWeight1 = Subnet_weight(self.kernel_size**2)
        self.moduleAlpha1 = Subnet_offset(self.kernel_size**2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size**2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size**2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size**2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size**2)
        self.moduleOcclusion = Subnet_occlusion()
        self.moduleBlend = Subnet_occlusion()

        self.generate_weight_subkernal = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(
                self.kernel_size ** 2, self.kernel_size ** 2
            ),
        )

        self.generate_weight_kernal = torch.nn.ModuleList([self.generate_weight_subkernal for _ in range(6 * 4)])
        self.generate_weight_softmax = torch.nn.Softmax(dim=1)

        self.module1by1_1 = torch.nn.Conv2d(
            in_channels=32, out_channels=4, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_3 = torch.nn.Conv2d(
            in_channels=128, out_channels=16, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_4 = torch.nn.Conv2d(
            in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=1
        )
        self.module1by1_5 = torch.nn.Conv2d(
            in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=1
        )

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        Weight1 = self.moduleWeight1(tensorCombine)
        Weight1_c1 = self.generate_weight_kernal[0](
            Weight1.transpose(3, 1)).transpose(3, 1)
        Weight1_c2 = self.generate_weight_kernal[1](
            Weight1.transpose(3, 1)).transpose(3, 1)
        Weight1_c3 = self.generate_weight_kernal[2](
            Weight1.transpose(3, 1)).transpose(3, 1)
        Weight1_classical = self.generate_weight_kernal[18](
            Weight1.transpose(3, 1)).transpose(3, 1)
        Weight1 = torch.cat([self.generate_weight_softmax(Weight1_c1), self.generate_weight_softmax(Weight1_c2),
                             self.generate_weight_softmax(Weight1_c3)], 1)


        Alpha1 = self.moduleAlpha1(tensorCombine)
        Alpha1_c1 = self.generate_weight_kernal[3](
            Alpha1.transpose(3, 1)).transpose(3, 1)
        Alpha1_c2 = self.generate_weight_kernal[4](
            Alpha1.transpose(3, 1)).transpose(3, 1)
        Alpha1_c3 = self.generate_weight_kernal[5](
            Alpha1.transpose(3, 1)).transpose(3, 1)
        Alpha1_classical = self.generate_weight_kernal[19](
            Alpha1.transpose(3, 1)).transpose(3, 1)
        Alpha1 = torch.cat([Alpha1_c1, Alpha1_c2, Alpha1_c3], 1)

        Beta1 = self.moduleBeta1(tensorCombine)
        Beta1_c1 = self.generate_weight_kernal[6](
            Beta1.transpose(3, 1)).transpose(3, 1)
        Beta1_c2 = self.generate_weight_kernal[7](
            Beta1.transpose(3, 1)).transpose(3, 1)
        Beta1_c3 = self.generate_weight_kernal[8](
            Beta1.transpose(3, 1)).transpose(3, 1)
        Beta1_classical = self.generate_weight_kernal[20](
            Beta1.transpose(3, 1)).transpose(3, 1)
        Beta1 = torch.cat([Beta1_c1, Beta1_c2, Beta1_c3], 1)

        Weight2 = self.moduleWeight2(tensorCombine)
        Weight2_c1 = self.generate_weight_kernal[9](
            Weight2.transpose(3, 1)).transpose(3, 1)
        Weight2_c2 = self.generate_weight_kernal[10](
            Weight2.transpose(3, 1)).transpose(3, 1)
        Weight2_c3 = self.generate_weight_kernal[11](
            Weight2.transpose(3, 1)).transpose(3, 1)
        Weight2_classical = self.generate_weight_kernal[21](
            Weight2.transpose(3, 1)).transpose(3, 1)
        Weight2 = torch.cat([self.generate_weight_softmax(Weight2_c1), self.generate_weight_softmax(Weight2_c2),
                             self.generate_weight_softmax(Weight2_c3)], 1)

        Alpha2 = self.moduleAlpha2(tensorCombine)
        Alpha2_c1 = self.generate_weight_kernal[12](
            Alpha2.transpose(3, 1)).transpose(3, 1)
        Alpha2_c2 = self.generate_weight_kernal[13](
            Alpha2.transpose(3, 1)).transpose(3, 1)
        Alpha2_c3 = self.generate_weight_kernal[14](
            Alpha2.transpose(3, 1)).transpose(3, 1)
        Alpha2_classical = self.generate_weight_kernal[22](
            Alpha2.transpose(3, 1)).transpose(3, 1)
        Alpha2 = torch.cat([Alpha2_c1, Alpha2_c2, Alpha2_c3], 1)

        Beta2 = self.moduleBeta2(tensorCombine)
        Beta2_c1 = self.generate_weight_kernal[15](
            Beta2.transpose(3, 1)).transpose(3, 1)
        Beta2_c2 = self.generate_weight_kernal[16](
            Beta2.transpose(3, 1)).transpose(3, 1)
        Beta2_c3 = self.generate_weight_kernal[17](
            Beta2.transpose(3, 1)).transpose(3, 1)
        Beta2_classical = self.generate_weight_kernal[23](
            Beta2.transpose(3, 1)).transpose(3, 1)
        Beta2 = torch.cat([Beta2_c1, Beta2_c2, Beta2_c3], 1)

        Occlusion = self.moduleOcclusion(tensorCombine)
        Blend = self.moduleBlend(tensorCombine)

        featConv1 = self.module1by1_1(tensorConv1)
        featConv2 = self.module1by1_2(tensorConv2)
        featConv3 = self.module1by1_3(tensorConv3)
        featConv4 = self.module1by1_4(tensorConv4)
        featConv5 = self.module1by1_5(tensorConv5)

        return (Weight1.contiguous(),
                Alpha1.contiguous(),
                Beta1.contiguous(),
                Weight2.contiguous(),
                Alpha2.contiguous(),
                Beta2.contiguous(),
                Weight1_classical.contiguous(),
                Alpha1_classical.contiguous(),
                Beta1_classical.contiguous(),
                Weight2_classical.contiguous(),
                Alpha2_classical.contiguous(),
                Beta2_classical.contiguous(),
                Occlusion,
                Blend,
                featConv1,
                featConv2,
                featConv3,
                featConv4,
                featConv5
                )

class Sep_AdaCoF(torch.nn.Module):
    def __init__(self, args):
        super(Sep_AdaCoF, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.channel_size = args.channel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation
        self.modulePad = torch.nn.ReplicationPad2d(
            [self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad]
        )
        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply
        self.AdaCoF = [self.moduleAdaCoF for _ in range(self.channel_size)]

        self.inpemb = input_embedding(1, self.args.d_model,self.args.max_len,self.args.device)
        self.layer = Channel_Attention_Layer(self.args.d_model, self.args.nheads, self.args.dim_feedforward, self.args.dropout)
        self.chn_att = Attention(self.layer, num_layers=self.args.num_fusion_layers)
        self.linear_output = nn.Linear(self.args.d_model, 1)

    def forward(self, frame, Weight, Alpha, Beta):
        N,C,H,W = frame.shape
        result = torch.zeros_like(frame)
        frame = self.modulePad(frame)
        for i in range(self.channel_size):
            result[:,i:i+1] = self.AdaCoF[i](frame[:,i:i+1].contiguous(), Weight[:,i*self.kernel_size**2:(i+1)*self.kernel_size**2].contiguous(), Alpha[:,i*self.kernel_size**2:(i+1)*self.kernel_size**2].contiguous(), Beta[:,i*self.kernel_size**2:(i+1)*self.kernel_size**2].contiguous(), self.dilation)

        # a = result.flatten(0,1)
        result = self.inpemb(result.flatten(-2,-1)[:,:,:,None])
        result = self.chn_att(result)
        result = self.linear_output(result)
        result = result.reshape(N,C,H,W)
        # b = result.flatten(0,1)
        # np.savez('feature.npz',input=a.cpu(),output=b.cpu())

        return result

class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.f(x)

class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.f(x)

class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LateralBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x
class FusionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x

class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96)):
        super(GridNet, self).__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert (
            len(grid_chs) == self.n_row
        ), "should give num channels for each row (scale stream)"

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(n_ch, n_ch))

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f"fusion_{r}_{c}", FusionBlock(n_ch*2, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"down_{r}_{c}", DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)
        self.register_parameter('param_add', nn.Parameter(torch.ones(10)))
        self.register_parameter('param_cat', nn.Parameter(torch.zeros(10)))

    def forward(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.param_cat[0]*self.fusion_1_0(torch.cat([self.down_0_1(state_01), self.lateral_1_0(state_10)], 1)) + self.param_add[0]*(self.down_0_1(
            state_01) + self.lateral_1_0(state_10))
        state_21 = self.param_cat[1]*self.fusion_2_0(torch.cat([self.down_1_1(state_11), self.lateral_2_0(state_20)], 1)) + self.param_add[1]*(self.down_1_1(
            state_11) + self.lateral_2_0(state_20))

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.param_cat[2]*self.fusion_1_1(torch.cat([self.down_0_2(state_02), self.lateral_1_1(state_11)], 1)) + self.param_add[2]*(self.down_0_2(
            state_02) + self.lateral_1_1(state_11))
        state_22 = self.param_cat[3]*self.fusion_2_1(torch.cat([self.down_1_2(state_12), self.lateral_2_1(state_21)], 1)) + self.param_add[3]*(self.down_1_2(
            state_12) + self.lateral_2_1(state_21))

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.param_cat[4]*self.fusion_1_2(torch.cat([self.up_1_0(state_23), self.lateral_1_2(state_12)], 1)) + self.param_add[4]*(self.up_1_0(
            state_23) + self.lateral_1_2(state_12))
        state_03 = self.param_cat[5]*self.fusion_0_2(torch.cat([self.up_0_0(state_13), self.lateral_0_2(state_02)], 1)) + self.param_add[5]*(self.up_0_0(
            state_13) + self.lateral_0_2(state_02))

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.param_cat[6]*self.fusion_1_3(torch.cat([self.up_1_1(state_24), self.lateral_1_3(state_13)], 1)) + self.param_add[6]*(self.up_1_1(
            state_24) + self.lateral_1_3(state_13))
        state_04 = self.param_cat[7]*self.fusion_0_3(torch.cat([self.up_0_1(state_14), self.lateral_0_3(state_03)], 1)) + self.param_add[7]*(self.up_0_1(
            state_14) + self.lateral_0_3(state_03))

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.param_cat[8]*self.fusion_1_4(torch.cat([self.up_1_2(state_25), self.lateral_1_4(state_14)], 1)) + self.param_add[8]*(self.up_1_2(
            state_25) + self.lateral_1_4(state_14))
        state_05 = self.param_cat[9]*self.fusion_0_4(torch.cat([self.up_0_2(state_15), self.lateral_0_4(state_04)], 1)) + self.param_add[9]*(self.up_0_2(
            state_15) + self.lateral_0_4(state_04))

        return self.lateral_final(state_05)


class TDNN(torch.nn.Module):
    def __init__(self, args):
        super(TDNN, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.context_synthesis = GridNet(
            254, 3
        )  # (in_channel, out_channel) = (126, 3) for the synthesis network

        self.modulePad = torch.nn.ReplicationPad2d(
            [self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad]
        )

        self.moduleAdaCoF = Sep_AdaCoF(self.args)
        self.moduleAdaCoF_classical = adacof_classical.FunctionAdaCoF.apply

    def forward(self, x):
        for i in range(x.shape[1] - 1):
            x_in_1 = torch.cat([x[:, i:i + 1], x[:, i + 1:i + 2]], dim=1)
            frame0, frame2 = x[:, i], x[:, i + 1]

            # np.savez('net_input.npz',frame0=frame0.cpu(),frame2=frame2.cpu())

            h0 = int(list(frame0.size())[2])
            w0 = int(list(frame0.size())[3])
            h2 = int(list(frame2.size())[2])
            w2 = int(list(frame2.size())[3])
            if h0 != h2 or w0 != w2:
                sys.exit("Frame sizes do not match")

            h_padded = False
            w_padded = False
            if h0 % 32 != 0:
                pad_h = 32 - (h0 % 32)
                frame0 = F.pad(frame0, (0, 0, 0, pad_h), mode="reflect")
                frame2 = F.pad(frame2, (0, 0, 0, pad_h), mode="reflect")
                h_padded = True

            if w0 % 32 != 0:
                pad_w = 32 - (w0 % 32)
                frame0 = F.pad(frame0, (0, pad_w, 0, 0), mode="reflect")
                frame2 = F.pad(frame2, (0, pad_w, 0, 0), mode="reflect")
                w_padded = True
            (Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, Weight1_classical,
                Alpha1_classical,Beta1_classical,Weight2_classical,Alpha2_classical,
                Beta2_classical,Occlusion,Blend,featConv1,featConv2,featConv3,featConv4,featConv5) = self.get_kernel(
                moduleNormalize(frame0), moduleNormalize(frame2)
            )

            tensorAdaCoF1 = (self.moduleAdaCoF(frame0, Weight1, Alpha1, Beta1) * 1.0)
            tensorAdaCoF2 = (self.moduleAdaCoF(frame2, Weight2, Alpha2, Beta2) * 1.0)

            # np.savez('feature2input.npz',tensorAdaCoF1=tensorAdaCoF1.cpu(),tensorAdaCoF2=tensorAdaCoF2.cpu(),featConv1=featConv1.cpu(),featConv2=featConv2.cpu(),featConv3=featConv3.cpu(),featConv4=featConv4.cpu(),featConv5=featConv5.cpu())

            frame1_warp = Occlusion * tensorAdaCoF1 + (1 - Occlusion) * tensorAdaCoF2

            # np.savez('main_arch_output.npz', frame1_warp=frame1_warp.cpu())

            w, h = self.modulePad(frame0).shape[2:]

            tensorConv1_ = F.interpolate(
                featConv1, size=(w, h), mode="bilinear", align_corners=False
            )
            tensorConv1L = (
                    self.moduleAdaCoF_classical(tensorConv1_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )
            tensorConv1R = (
                    self.moduleAdaCoF_classical(tensorConv1_, Weight2_classical, Alpha2_classical, Beta2_classical, self.dilation) * 1.0
            )

            tensorConv2_ = F.interpolate(
                featConv2, size=(w, h), mode="bilinear", align_corners=False
            )
            tensorConv2L = (
                    self.moduleAdaCoF_classical(tensorConv2_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )
            tensorConv2R = (
                    self.moduleAdaCoF_classical(tensorConv2_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )

            tensorConv3_ = F.interpolate(
                featConv3, size=(w, h), mode="bilinear", align_corners=False
            )
            tensorConv3L = (
                    self.moduleAdaCoF_classical(tensorConv3_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )
            tensorConv3R = (
                    self.moduleAdaCoF_classical(tensorConv3_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )

            tensorConv4_ = F.interpolate(
                featConv4, size=(w, h), mode="bilinear", align_corners=False
            )
            tensorConv4L = (
                    self.moduleAdaCoF_classical(tensorConv4_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )
            tensorConv4R = (
                    self.moduleAdaCoF_classical(tensorConv4_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )

            tensorConv5_ = F.interpolate(
                featConv5, size=(w, h), mode="bilinear", align_corners=False
            )
            tensorConv5L = (
                    self.moduleAdaCoF_classical(tensorConv5_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )
            tensorConv5R = (
                    self.moduleAdaCoF_classical(tensorConv5_, Weight1_classical, Alpha1_classical, Beta1_classical, self.dilation) * 1.0
            )

            tensorCombined = torch.cat(
                [
                    tensorAdaCoF1,
                    tensorAdaCoF2,
                    tensorConv1L,
                    tensorConv1R,
                    tensorConv2L,
                    tensorConv2R,
                    tensorConv3L,
                    tensorConv3R,
                    tensorConv4L,
                    tensorConv4R,
                    tensorConv5L,
                    tensorConv5R,
                ],
                dim=1,
            )

            frame1_feat = self.context_synthesis(tensorCombined)

            # np.savez('synth_output.npz', frame1_feat=frame1_feat.cpu())

            # np.savez('feature2output.npz', frame1_feat=frame1_feat.cpu())

            frame1 = (
                    Blend * frame1_feat + (1 - Blend) * frame1_warp
            )  # blending of the feature warp and ordinary warp

            if h_padded:
                frame1 = frame1[:, :, 0:h0, :]
            if w_padded:
                frame1 = frame1[:, :, :, 0:w0]

            # np.savez('net_output.npz', frame1=frame1.cpu())
            print('Temp stop')
            time.sleep(30)
            print('Start')

            if i == 0:
                result = torch.cat([x_in_1[:, 0][:, None], frame1[:, None], x_in_1[:, 1][:, None]], 1)
            else:
                result = torch.cat([result, frame1[:, None], x_in_1[:, 1][:, None]], 1)
        return result
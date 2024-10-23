import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from math import exp

def image_lr_normalization_maxmin(data):
    L, N, C, H, W = data.shape
    data_orin = data
    data =data.permute(2,1,0,3,4).flatten(start_dim=1,end_dim=4) 

    cmax_list, cmin_list = [], []
    for i in range(C):
        simage = np.array(data[i])
        s_max,s_min = max(simage), min(simage)
        cmax_list.append(s_max)
        cmin_list.append(s_min)
    cmax, cmin = np.array(cmax_list)[:,None], np.array(cmin_list)[:,None]
    max_min = np.concatenate([cmax, cmin], 1)
    del data

    max_adj, min_adj = np.zeros((L, N, C, H, W )), np.zeros((L, N, C, H, W ))
    for k in range(L):
        for i in range(N):
            for j in range(C):
                max_adj[k, i, j] = np.full((1, 1, 1, H, W), max_min[j, 0])
                min_adj[k, i, j] = np.full((1, 1, 1, H, W), max_min[j, 1])
    data_norm = (data_orin-min_adj)/(max_adj-min_adj)
    return data_norm

L1 = nn.L1Loss()

L2 = nn.MSELoss()

def WRMSE(x,y):
    return torch.sqrt(L2(x,y))

def WACC(y_pred,y_true,arrshape):
    # N, T, C, H, W
    y_pred_o, y_true_o = y_pred.reshape(arrshape),y_true.reshape(arrshape)
    ACC = np.empty([y_pred_o.shape[1]])
    for i in range(y_pred_o.shape[1]):
        y_pred, y_true = y_pred_o[:,i], y_true_o[:,i]
        clim = y_true.mean(0)
        a = y_true - clim
        a_prime = (a - a.mean())
        fa = y_pred - clim
        fa_prime = (fa - fa.mean())
        ACC[i] = (
                torch.sum(fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(fa_prime ** 2) * torch.sum(a_prime ** 2)
                )
        )
    return ACC.mean()

def WACC_Sub(y_pred,y_true):
    clim = y_true.mean(0)
    a = y_true - clim
    a_prime = (a - a.mean())
    fa = y_pred - clim
    fa_prime = (fa - fa.mean())
    ACC = (
            torch.sum(fa_prime * a_prime) /
            torch.sqrt(
                torch.sum(fa_prime ** 2) * torch.sum(a_prime ** 2)
            )
    )
    return ACC.mean()

def PSNR(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return torch.tensor(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = img1 * 0.5 + 0.5
    img2 = img2 * 0.5 + 0.5

    if len(img1.size()) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.size()) == 3:
        img2 = img2.unsqueeze(0)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

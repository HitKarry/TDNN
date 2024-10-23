import torch

class Configs:
    def __init__(self):
        pass

configs = Configs()

configs.gpu_ids = 0
configs.device = torch.device('cuda:'+str(configs.gpu_ids))
configs.batch_size_test = 4


configs.max_len = 3
configs.channel_size = 3
configs.kernel_size = 25
configs.dilation = 2
configs.patch_size = [8,8]
configs.d_model = 256
configs.nheads = 4
configs.dim_feedforward = 512
configs.dropout = 0.2
configs.num_fusion_layers = 3



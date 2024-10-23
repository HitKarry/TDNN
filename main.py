import torch
import numpy as np
from torch.utils.data import Dataset
from models.network import TDNN
from config import configs
import math
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datetime
import torch.optim as optim

ssl._create_default_https_context = ssl._create_unverified_context
torch.cuda.set_device('cuda:'+str(configs.gpu_ids))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class dataset_generator(Dataset):
    def __init__(self, data_lr, data_hr):
        super().__init__()

        self.data_lr = data_lr
        self.data_hr = data_hr
        assert self.data_lr.shape[0] == self.data_hr.shape[0]

    def GetDataShape(self):
        return {'data_lr': self.data_lr.shape,
                'data_hr': self.data_hr.shape}

    def __len__(self,):
        return self.data_lr.shape[0]

    def __getitem__(self, idx):
        return self.data_lr[idx], self.data_hr[idx]

if __name__ == '__main__':
    print(configs.__dict__)

    print('\nreading data')
    data = np.load('../data/DATA_36h.npz')
    data_lr = torch.tensor(data['x'])
    data_hr = torch.tensor(data['y'])

    print('processing dataset')
    dataset_eval = dataset_generator(data_lr, data_hr)
    print(dataset_eval.GetDataShape())

    del data_lr

    model = TDNN(configs).to(configs.device)
    net = torch.load('checkpoint.chk')
    model.load_state_dict(net['net'])
    model.eval()

    data = DataLoader(dataset_eval, configs.batch_size_test, shuffle=False)

    starttime = datetime.datetime.now()
    with torch.no_grad():
        for i,(lr,hr) in enumerate(data):
            pred_temp = model(lr.float().to(configs.device))
            if i == 0:
                pred = pred_temp
                label = hr
            else:
                pred = torch.cat((pred, pred_temp), 0)
                label = torch.cat((label, hr), 0)
    endtime = datetime.datetime.now()
    print('SPEND TIME:', endtime - starttime)

    np.savez('result.npz', sr=pred.cpu(), hr=label.cpu())
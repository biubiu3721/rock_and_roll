import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        in_dim = 7
        out_dim = 1
        hd1_dim = 7
        hd2_dim = 7
        self.model = nn.Sequential(nn.Linear(in_dim, out_dim))
        #self.fc2 = nn.Linear(hd1_dim, out_dim)
        # self.fc3 = nn.Linear(256, 512)
        # self.fc4 = nn.Linear(512, 1)
        # self.fc5 = nn.Linear(1024, 512)
        # self.fc6 = nn.Linear(512, 256)
        # self.fc7 = nn.Linear(256, 1)
        # self.bn1 = torch.nn.BatchNorm1d(hd1_dim,
        #                                 eps=1e-05,
        #                                 momentum=0.1,
        #                                 affine=True,
        #                                 track_running_stats=True,
        #                                 device=None,
        #                                 dtype=None)
        # self.bn2 = torch.nn.BatchNorm1d(1,
        #                                 eps=1e-05,
        #                                 momentum=0.1,
        #                                 affine=True,
        #                                 track_running_stats=True,
        #                                 device=None,
        #                                 dtype=None)
        # self.bn3 = torch.nn.BatchNorm1d(512,
        #                                 eps=1e-05,
        #                                 momentum=0.1,
        #                                 affine=True,
        #                                 track_running_stats=True,
        #                                 device=None,
        #                                 dtype=None)
    def forward(self, x):
        x1 = self.model(x)
        #x = self.fc2(x)
        #x = torch.sigmoid(self.bn2(self.fc2(x)))
        # x = torch.sigmoid(self.bn3(self.fc3(x)))
        # x = self.fc4(x)
        # x = self.fc5(x)
        # x = self.fc6(x)
        # x = self.fc7(x)
        return x1



import torch
import torch.nn as nn
from .SSRP_MS import SSRP_MS
from .WavKAN import WavKANLinear
from .PCAw_Pool import PCAw_Pool

class CNN_PCAw_SSRPMS_KAN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_PCAw_SSRPMS_KAN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(1, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            PCAw_Pool(kernel_size=(3, 3), stride=(3, 3)),
        )

        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.ssrp_ms = SSRP_MS(base_window=3, num_levels=5)
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(1024, 128)
        self.kan = WavKANLinear(128, num_classes)

    def forward(self, x):
        return self.kan(self.fc(self.flatten(self.ssrp_ms(self.conv3(self.conv2(self.conv1(x)))))))

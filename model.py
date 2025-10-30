import numpy as np
from torch.nn import init
import torch.nn as nn
import torch

# whisper

# audio classification
class AudioClassification(nn.Module):
    def __init__(self, C = 2, H = 64, W = 64):
        super().__init__()
        conv_layers = []
        
        # first convolution block
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.ln1 = nn.LayerNorm([8, 32, 32])
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu', mode="fan_in", a = 0)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.ln1]
         
        # second convolution block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.ln2 = nn.LayerNorm([16, 16, 16])
        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu", mode="fan_in", a = 0)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.ln2]
        
        # third convolution block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.ln3 = nn.LayerNorm([32, 8, 8])
        init.kaiming_normal_(self.conv3.weight, nonlinearity="relu", mode="fan_in", a = 0)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.ln3]
        
        # fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.LayerNorm([64, 4, 4])
        init.kaiming_normal_(self.conv4.weight, nonlinearity="relu", mode="fan_in", a = 0)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]
        
        # Bổ sung lớp dropout
        self.dropout = nn.Dropout(p=0.3)
        
        #linear
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=3)
        
        #wrap
        self.conv = nn.Sequential(*conv_layers)
        
    def forward(self, x):
        x = self.conv(x)
        # print(x.shape) # Trả ra đầu ra dạng [batch_size, 64, 4, 4]
        
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.lin(x)
        return x

if __name__== "__main__":
    model = AudioClassification().to("cuda")
    input = torch.rand((5, 2, 64, 64)).to("cuda")
    result = model(input)
    from torchsummary import summary
    from torchvision import models
    summary(model, (2, 64, 64))
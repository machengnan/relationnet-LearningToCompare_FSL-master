import torch.nn as nn
from torchvision import models
from torch.hub import  load_state_dict_from_url
import torch
import torch.nn.functional as F

model_urls = {
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}

class shuffle_relationnet(models.ShuffleNetV2):
    def __init__(self, input_size, hidden_size, pretrained = True):

        super(shuffle_relationnet, self).__init__(stages_repeats=[4, 8, 4], stages_out_channels =  [24, 116, 232, 464, 1024])
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['shufflenetv2_x1.0'],
                                                  progress=True)
            self.load_state_dict(state_dict)

        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(int(input_size*4),hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    global x
    def forward(self, x):
        # print(x.size())
        x = self.layer1(x)
        x = x.view(-1,3,15,15)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.layer2(x)
        # print(x.size())
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # print(x.size())
        # x = self.fc1(x)
        # x = x.mean([2, 3])
        return x

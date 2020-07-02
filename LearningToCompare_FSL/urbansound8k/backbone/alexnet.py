import torch
import torch.nn as nn
from torchvision import models
from torch.hub import  load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}

class Alexnet(models.AlexNet):
    def __init__(self,last_channel= 1600,pretrained=True):

        super(Alexnet, self).__init__()

        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                  progress=True)
            self.load_state_dict(state_dict)
        self.features[0] = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.last_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, last_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        return x.view(x.size(0), -1)

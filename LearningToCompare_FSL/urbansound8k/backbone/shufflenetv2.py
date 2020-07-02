import torch.nn as nn
from torchvision import models
from torch.hub import  load_state_dict_from_url


model_urls = {
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}

class shufflenet(models.ShuffleNetV2):
    def __init__(self, pretrained = True):

        super(shufflenet, self).__init__(stages_repeats=[4, 8, 4], stages_out_channels =  [24, 116, 232, 464, 1024])
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['shufflenetv2_x1.0'],
                                                  progress=True)
            self.load_state_dict(state_dict)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 1600, kernel_size=1, padding=0),
            nn.BatchNorm2d(1600),
            nn.ReLU()
        )

    global x
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.layer1(x)
        # print(x.size())
        # x = self.fc1(x)
        # x = x.mean([2, 3])
        return x


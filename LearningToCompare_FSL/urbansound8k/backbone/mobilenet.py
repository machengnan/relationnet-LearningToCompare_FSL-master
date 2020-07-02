import torch.nn as nn
from torchvision import models
from torch.hub import  load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
}


class mobilenet(models.mobilenet.MobileNetV2):
    def __init__(self, pretrained = True):

        super(mobilenet, self).__init__()
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=True)
            self.load_state_dict(state_dict)
            self.out_channels = 1600

        self.layer1 = nn.Sequential(
                        nn.Conv2d(1280,1600,kernel_size=1,padding=0),
                        nn.BatchNorm2d(1600),
                        nn.ReLU()
        )


        # self.fc1 = nn.Linear(576000, 80000)

# 5*10*64*5*5
    global x
    def forward(self, x):
        x = x.contiguous()
        x = self.features(x)
        x = self.layer1(x)
        # x = self.fc1(x)
        # return x.view(x.size(0), -1)
        # print(x.size())
        return x


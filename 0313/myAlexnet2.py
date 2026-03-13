# 作者: 宇亮
# 2026年03月13日16时50分10秒
# Julian_guo153@qq.com

# from torchvision import models
# model = models.alexnet()
# print(model)

import torch
import torch.nn as nn


class myModel(nn.Module):
    def __init__(self, num_classes):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool4 = nn.AdaptiveMaxPool2d(6)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.pool4(x) # batch*256*6*6
        x = x.view(x.size()[0], -1) # batch*9216
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x




if __name__ == '__main__':
    model = myModel(1000)
    data = torch.ones((4, 3, 224, 224))
    pred = model(data)
    pass

import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, pretrained:bool):
        super(Resnet, self).__init__()
        if pretrained:    
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            print("Loading weights version 'IMAGENET1K_V2'")
        else:
            weights = None
        self.my_model = models.resnet50(weights = weights)

        self.fc = nn.Linear(1000, 2)
        nn.init.xavier_normal_(self.fc.weight.data)
        self.fc.bias.data.zero_()

    def forward(self, x):
        self.resnet = self.my_model(x)
        x = self.fc(self.resnet)
        return x


# ======Densenet
class Densenet(nn.Module):
    def __init__(self, pretrained:bool, local_tuning = False):
        super(Densenet, self).__init__()
        if pretrained:    
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            print("Loading weights version 'IMAGENET1K_V1'")
        else:
            weights = None
        self.my_model = models.densenet121(weights = weights)

        self.fc = nn.Linear(1000, 2)
        nn.init.xavier_normal_(self.fc.weight.data)
        self.fc.bias.data.zero_()

    def forward(self, x):
        self.resnet = self.my_model(x)
        x = self.fc(self.resnet)
        return x

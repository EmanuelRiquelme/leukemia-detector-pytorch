import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3,EfficientNet_B3_Weights
class Model(nn.Module):
    def __init__(self,num_classes = 2):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.model = self.__load_model__()

    def __load_model__(self):
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1536,self.num_classes)
        )
        return model

    def forward(self,input_data):
        return self.model(input_data)

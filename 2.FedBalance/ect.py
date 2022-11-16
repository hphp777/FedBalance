import torch
import torch.nn as nn
import torchvision.models as models 
from torchsummary import summary

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = ResNet50.resnet56() ####
# model = ResNet50_fedalign.resnet56()
model = models.efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)
model.to(device)

# summary(model, (3, 224, 224))
print(model.features[7]) # last block
import torch
import torch.nn as nn
import torchvision.models as models

def ResNet50(out_features = 100):
    return getattr(models, "resnet50")(pretrained=False, num_classes = out_features)
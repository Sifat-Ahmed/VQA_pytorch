import torch
import torch.nn as nn
import torchvision.models as models

def ResNet50(out_features = 10):
    return getattr(models, "resnet152")(pretrained=False, num_classes = out_features)
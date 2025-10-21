import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model, num_classes, feature_extract, use_pretrained=True):
  model_ft = None
  input_size = 0

  if model == "resnet":
    if use_pretrained:
            model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model_ft = models.resnet18(weights=None)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model == "alexnet":
    if use_pretrained:
        model_ft = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    else:
        model_ft = models.alexnet(weights=None)
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

  elif model == "vgg":
    if use_pretrained:
        model_ft = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
    else:
        model_ft = models.vgg11_bn(weights=None)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

  return model_ft, input_size

def get_params_to_update(model, feature_extract):
    params_to_update = model.parameters()

    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

    return params_to_update
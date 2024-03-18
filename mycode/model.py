import torch
import torchvision

def resnet34(n_categories, cuda=False):
    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    # replace final fc layer and initialize new layer's weights
    model.fc = torch.nn.Linear(model.fc.in_features, n_categories)
    torch.nn.init.xavier_uniform_(model.fc.weight)
    if cuda:
        return model.cuda()
    return model

def load_model(model_path):
    model = torch.load(model_path)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    return model
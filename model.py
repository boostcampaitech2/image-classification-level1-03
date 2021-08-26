import torch
import torch.nn as nn
import torchvision


def Model_M():
    model = torchvision.models.resnet50(pretrained=True)
    # print("네트워크 필요 입력 채널 개수", model.conv1.weight.shape[1])
    # print("네트워크 출력 채널 개수 (예측 class type 개수)", model.fc.weight.shape)
    # print(model)
    in_features = model.fc.weight.shape[1]
    model.fc = nn.Linear(in_features=in_features, out_features=3, bias=True)
    # torch.nn.init.xavier_uniform_(model.fc.weight)
    # stdv = 1 / math.sqrt(model.fc.weight.size(1))
    # model.fc.bias.data.uniform_(-stdv, stdv)

    return model


def Model_G():
    model = torchvision.models.resnet50(pretrained=True)
    in_features = model.fc.weight.shape[1]
    # model.fc = nn.Linear(in_features=in_features, out_features=3, bias=True)
    # model.load_state_dict(torch.load("model_M.pth"))
    model.fc = nn.Linear(in_features=in_features, out_features=2, bias=True)

    return model


def Model_A():
    model = torchvision.models.resnet50(pretrained=True)
    in_features = model.fc.weight.shape[1]
    # model.fc = nn.Linear(in_features=in_features, out_features=2, bias=True)
    # model.load_state_dict(torch.load("model_G.pth"))
    model.fc = nn.Linear(in_features=in_features, out_features=3, bias=True)

    return model
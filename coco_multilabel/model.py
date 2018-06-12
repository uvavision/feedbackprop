import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.utils

class CocoMultilabelModel(nn.Module):
    """ multi-label model for COCO words """
    def __init__(self, args, n_categories):
        super(CocoMultilabelModel, self).__init__()

        self.n_categories = n_categories
        self.base_network = models.resnet18(pretrained = True)

        if not args.finetune:
            for param in self.base_network.parameters():
                param.requires_grad = False

        self.base_network.fc = nn.Conv2d(512, n_categories, 1)
        print('initializing network Resnet 18', self.n_categories)

    def forward(self, image):

        x = self.base_network.conv1(image)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)
        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)

        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)


        # Skipping an avg-pool layer from Resnet18.
        x = self.base_network.fc(x).sigmoid()

        # Take the max for each prediction map.
        return x.max(dim = 2, keepdim=True)[0].max(dim = 3, keepdim=True)[0].squeeze()


if __name__=='__main__':
    # Test code for the label augmented model.
    model = CocoMultilabelModel(3)
    pad = model.paddingToken

    image = Variable(torch.randn(2, 3, 224, 224))
    labels = Variable(torch.LongTensor([[0, 1, 2, pad, pad, pad], [3, 4, pad, pad, pad, pad]]))
    label_lengths = [3, 2]
    outputs = model(image, labels, label_lengths)
    print(outputs.size())
    print(outputs)

    image = Variable(torch.randn(3, 3, 224, 224))
    labels = Variable(torch.LongTensor([[0, 1, 2, pad, pad, pad],
                                        [3, 4, pad, pad, pad, pad],
                                        [pad, pad, pad, pad, pad, pad]]))
    label_lengths = [3, 2, 0]
    outputs = model(image, labels, label_lengths)
    print(outputs.size())
    print(outputs)

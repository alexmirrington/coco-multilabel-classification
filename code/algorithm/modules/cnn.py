"""Module containing code for a CNN object detector."""
import torch
import torchvision.models as models
from torch.nn import Linear, Module, Sigmoid


class CNN(Module):
    """A CNN model."""

    def __init__(self, classes, model_name, threshold=0.5):
        """Initialise the CNN, freezing its parameters to avoid \
        additional training."""
        super().__init__()
        if model_name == 'resnet':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
        elif model_name == 'squeezenet':
            self.model = models.squeezenet1_0(pretrained=True)
        elif model_name == 'vgg':
            self.model = models.vgg16(pretrained=True)
        elif model_name == 'densenet':
            self.model = models.densenet161(pretrained=True)
        elif model_name == 'inception':
            self.model = models.inception_v3(pretrained=True)
        elif model_name == 'googlenet':
            self.model = models.googlenet(pretrained=True)
        elif model_name == 'shufflenet':
            self.model = models.shufflenet_v2_x1_0(pretrained=True)
        elif model_name == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
        elif model_name == 'resnext':
            self.model = models.resnext50_32x4d(pretrained=True)
        elif model_name == 'wide_resnet':
            self.model = models.wide_resnet50_2(pretrained=True)
        elif model_name == 'mnasnet':
            self.model = models.mnasnet1_0(pretrained=True)
        else:
            self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear = Linear(1000, len(classes))
        self.activation = Sigmoid()
        self.threshold = threshold
        self.classes = classes

    def forward(self, images, captions, device):
        """Propagate data through the model."""
        input_batch = torch.stack(images)
        self.model.eval()
        results = self.model(input_batch)
        output = self.linear(results)
        preds = self.activation(output) > self.threshold
        return preds, output

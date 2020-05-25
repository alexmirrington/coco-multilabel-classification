"""Module containing code for a FasterRCNN object detector."""
from numpy import unique
from torch.nn import Module
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterRCNN(Module):
    """A FasterRCNN model, pretrained on COCO train2017."""

    def __init__(self):
        """Initialise the RCNN, freezing its parameters to avoid \
        additional training."""
        super().__init__()
        self.rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
        for param in self.rcnn.parameters():
            param.requires_grad = False

    def forward(self, *data):
        """Propagate data through the model."""
        images, _ = data
        results = self.rcnn(images)
        labels = [unique(result['labels'].cpu().numpy()) for result in results]
        labels = [[lbl for lbl in lbls if lbl < 20] for lbls in labels]
        return tuple(labels), results

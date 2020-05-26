"""Module containing code for a FasterRCNN object detector."""
import torch
from torch.nn import Linear, Module, Sigmoid
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utilities import binarise_labels


class FasterRCNN(Module):
    """A FasterRCNN model, pretrained on COCO train2017."""

    def __init__(self, classes, threshold=0.5):
        """Initialise the RCNN, freezing its parameters to avoid \
        additional training."""
        super().__init__()
        self.rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
        for param in self.rcnn.parameters():
            param.requires_grad = False
        self.linear = Linear(len(classes), len(classes))
        self.activation = Sigmoid()
        self.threshold = threshold
        self.classes = classes

    def forward(self, images, captions, device):
        """Propagate data through the model."""
        self.rcnn.eval()
        results = self.rcnn(images)
        # Sort scores low to high so lbl2score contains highest scores
        labels = [reversed(result['labels'].cpu().numpy())
                  for result in results]
        scores = [reversed(result['scores'].cpu().numpy())
                  for result in results]
        lbl2score = [list(zip(ls, ss)) for ls, ss in zip(labels, scores)]
        lbl2score = [{lbl: s for lbl, s in l2s if lbl in self.classes}
                     for l2s in lbl2score]
        labels = [list(l2s.keys()) for l2s in lbl2score]
        labels, _ = binarise_labels(labels, self.classes)
        # Convert lbl2score to matrix
        score_mat = [[l2s.get(i, 0) for i in self.classes]for l2s in lbl2score]
        score_mat = torch.Tensor(score_mat).to(device)
        # Learn linear weights on score matrix
        output = self.linear(score_mat)
        preds = self.activation(output) > self.threshold
        return preds, output

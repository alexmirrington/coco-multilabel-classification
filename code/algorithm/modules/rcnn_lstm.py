"""Module contaning a Faster RCNN model combined with an LSTM."""
import torch
import torch.nn as nn
from modules.faster_rcnn import FasterRCNN
from modules.lstm import BiLSTM


class RCNN_LSTM(nn.Module):
    """Predictions of RCNN and LSTM model combined.

    The scores from the image model (RCNN) and text model (LSTM)
    are concatenated and passed through a linear layer to compute
    final scores.
    """

    def __init__(self, classes, embeddings_dim=100, threshold=0.5):
        """Initialise the combined model."""
        super(RCNN_LSTM, self).__init__()

        self.txt_model = BiLSTM(classes)
        self.img_model = FasterRCNN(classes)
        self.combine = nn.Linear(len(classes)*2, len(classes))
        self.threshold = threshold
        self.classes = classes

    def forward(self, images, captions, device):
        """Propagate data through the model."""
        _, img_scores = self.img_model(images, captions, device)
        _, txt_scores = self.txt_model(images, captions, device)

        combined_scores = self.combine(torch.cat((img_scores, txt_scores),
                                                 dim=1))

        preds = torch.sigmoid(combined_scores) > self.threshold
        return preds, combined_scores


class RCNN_LSTM_Bilinear(nn.Module):
    """Predictions of RCNN and LSTM model combined.

    The scores from the image model (RCNN) and text model (LSTM)
    are passed through a bilinear layer to compute the final scores.
    """

    def __init__(self, classes, embeddings_dim=100, threshold=0.5):
        """Initialise the combined model."""
        super(RCNN_LSTM_Bilinear, self).__init__()

        self.txt_model = BiLSTM(classes)
        self.img_model = FasterRCNN(classes)
        self.combine = nn.Bilinear(len(classes), len(classes), len(classes))
        self.threshold = threshold
        self.classes = classes

    def forward(self, images, captions, device):
        """Propagate data through the model."""
        _, img_scores = self.img_model(images, captions, device)
        _, txt_scores = self.txt_model(images, captions, device)

        combined_scores = self.combine(img_scores, txt_scores)

        preds = torch.sigmoid(combined_scores) > self.threshold
        return preds, combined_scores

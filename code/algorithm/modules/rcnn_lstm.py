import torch
import torch.nn as nn


from modules.lstm import BiLSTM
from modules.faster_rcnn import FasterRCNN



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

        combined_scores = self.combine(torch.cat((img_scores, txt_scores), dim=1))

        preds = torch.sigmoid(combined_scores) > self.threshold
        return preds, combined_scores

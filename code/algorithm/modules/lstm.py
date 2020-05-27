import re
import torch.nn as nn
import torch


class BiLSTM(nn.Module):
    """BiLSTM which predicts multiple labels from caption data."""

    def __init__(self, classes, embeddings_dim=100, hidden_size=128, threshold=0.5):
        """Initialise the BiLSTM.

        Args
        ---
        `embeddings_dim`: dimension of the embeddings used for captions
        `n_classes`: number of classes to predict
        `hidden_size`: size of hidden weights of the LSTM in each direction
        `threshold`: threshold for the sigmoid of activations to reach to be
                        output as a predicted class
        """
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embeddings_dim, hidden_size, batch_first=True, bidirectional=True)

        # Size of linear input is hidden_size*2 since it takes in concatenation of hidden states.
        self.linear = nn.Linear(hidden_size*2, len(classes))
        self.threshold = threshold


    def forward(self, images, captions, device):
        """Propagate data through the model.

        Note that the images are not used by the BiLSTM, but are accepted to fit
        the signature of other models.

        Args
        ---
        `images`: tuple of images, where each image is a torch tensor
                    (as output by Dataset/DataLoader)
        `captions`: tuple of captions, where each caption is a list of embedding tensors
                        (as output by Dataset/DataLoader)
        `device`: torch device the model runs on
        """

        seq_lens = torch.LongTensor(list(map(len, captions)))

        # Tensor to store padded sequences
        # shape is (batch_size, max_seq_len, embedding_dim)
        padded_captions = torch.zeros(len(captions), seq_lens.max(), captions[0][0].numel())

        for i, caption in enumerate(captions):
            caption_len = seq_lens[i]
            padded_captions[i, :caption_len] = torch.stack(caption)

        # Pack the embedding sequence for input into the model using
        # https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        packed_captions = torch.nn.utils.rnn.pack_padded_sequence(padded_captions,
                                               seq_lens.cpu().numpy(),
                                               batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed_captions)

        #Concatenate hidden states
        hidden_concat = torch.cat((h_n[0,:,:], h_n[1,:,:]), dim=1)

        scores = self.linear(hidden_concat)
        preds = torch.sigmoid(scores) > self.threshold
        return preds, scores


# class LSTM(nn.Module):
#     """LSTM which predicts multiple labels from caption data."""
#
#     def __init__(self, embeddings_dim, n_classes=20, hidden_size=128, threshold=0.5):
#         """Initialise the BiLSTM.
#
#         Args
#         ---
#         `embeddings_dim`: dimension of the embeddings used for captions
#         `n_classes`: number of classes to predict
#         `hidden_size`: size of hidden weights of the LSTM
#         `threshold`: threshold for the sigmoid of activations to reach to be
#                         output as a predicted class
#         """
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(embeddings_dim, hidden_size, batch_first=True)
#
#         # Size of linear input is hidden_size*2 since it takes in concatenation of hidden states.
#         self.linear = nn.Linear(hidden_size, n_classes)
#         self.threshold = threshold
#
#
#     def forward(self, images, captions, device):
#         """Propagate data through the model.
#
#         Note that the images are not used by the LSTM, but are accepted to fit
#         the signature of other models.
#
#         Args
#         ---
#         `images`: tuple of images, where each image is a torch tensor
#                     (as output by Dataset/DataLoader)
#         `captions`: tuple of captions, where each caption is a list of embedding tensors
#                         (as output by Dataset/DataLoader)
#         `device`: torch device the model runs on
#         """
#
#         seq_lens = torch.LongTensor(list(map(len, captions)))
#
#         # Tensor to store padded sequences
#         # shape is (batch_size, max_seq_len, embedding_dim)
#         padded_captions = torch.zeros(len(captions), seq_lens.max(), captions[0][0].numel())
#
#         for i, caption in enumerate(captions):
#             caption_len = seq_lens[i]
#             padded_captions[i, :caption_len] = torch.stack(caption)
#
#         # Pack the embedding sequence for input into the model using
#         # https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
#         packed_captions = torch.nn.utils.rnn.pack_padded_sequence(padded_captions,
#                                                seq_lens.cpu().numpy(),
#                                                batch_first=True, enforce_sorted=False)
#         _, (h_n, c_n) = self.lstm(packed_captions)
#
#
#         scores = self.linear(h_n.squeeze())
#         preds = torch.sigmoid(scores) > self.threshold
#         return preds, scores

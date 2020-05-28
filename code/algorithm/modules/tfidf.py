"""Module containing TFIDF language based model."""
import torch
import torch.nn as nn

class TFIDF(nn.Module):
    """TFIDF vector for captions is passed through a linear layer to predict."""

    def __init__(self, classes, tfidf_vectorizer, threshold=0.5):
        """Initialise the TFIDF model."""
        super(TFIDF, self).__init__()
        self.tfidf_vectorizer = tfidf_vectorizer
        self.vocab_size = len(tfidf_vectorizer.vocabulary_)
        self.linear = nn.Linear(self.vocab_size, len(classes))
        self.threshold = threshold

    def forward(self, images, captions, device):
        """Propagate data through the model."""
        # Get the tfidf vectors for each caption
        tfidfs = torch.Tensor(self.tfidf(captions))

        # Pass through linear layer
        scores = self.linear(tfidfs)

        # Calculate predictions using threshold
        preds = torch.sigmoid(scores) > self.threshold
        return preds, scores

    def tfidf(self, captions):
        return self.tfidf_vectorizer.transform(captions).toarray()

""" Preprocessing utilities."""
import re

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_caption(caption, rm_stopwords=True, stem=False):
    """Preprocess the given caption.

    Params
    ---
    `caption`: The input caption as a string
    `rm_stopwords`: If True, remove stopwords from the caption when tokenising

    Returns
    ---
    List of tokens extracted from the caption.
    """
    # Convert to lowercase
    caption = caption.lower()

    # Remove punctuation
    caption = re.sub(r'[^\w\s]', '', caption)

    # Tokenise
    caption = word_tokenize(caption)

    # Remove stopwords
    if rm_stopwords:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        caption = [w for w in caption if w not in stopwords]

    # Stemming
    if stem:
        stemmer = SnowballStemmer('english')
        caption = [stemmer.stem(w) for w in caption]

    return caption

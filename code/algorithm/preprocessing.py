import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_caption(caption, rm_stopwords=True):
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

    if rm_stopwords:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        # Uncomment the below line to prevent removal of certain stopwords.
        # stopwords = stopwords.difference(set(['him', 'her']))
        caption = [w for w in caption if not w in stopwords]

    return caption

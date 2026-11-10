"""
preprocess.py
-------------
Text preprocessing for the Fake News Detection project.

Steps performed (in order):
    1. Convert text to lowercase
    2. Remove HTML tags and URLs
    3. Remove punctuation, digits and other special characters
    4. Tokenization (split the text into words)
    5. Stopword removal (remove common words like "the", "is", "and")

No external download is needed: the stopword list is written below,
so this file works offline on any laptop.
"""

import re

# A common list of English stopwords (words that carry little meaning).
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "d", "did", "do", "does",
    "doing", "don", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "ll", "m", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "she", "should", "so",
    "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
}


def clean_text(text):
    """Lowercase the text and remove URLs, HTML tags, punctuation and digits."""
    text = str(text).lower()                          # 1. lowercase
    text = re.sub(r"<.*?>", " ", text)                # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)     # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)             # remove punctuation, digits, symbols
    text = re.sub(r"\s+", " ", text).strip()          # remove extra spaces
    return text


def tokenize(text):
    """Split a cleaned sentence into a list of words (tokens)."""
    return text.split()


def remove_stopwords(tokens):
    """Remove stopwords and one-letter tokens from a list of tokens."""
    return [word for word in tokens if word not in STOPWORDS and len(word) > 1]


def preprocess_text(text):
    """
    Full pipeline: clean -> tokenize -> remove stopwords.
    Returns a single string so it can be given directly to TfidfVectorizer.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    # Quick demo: python preprocess.py
    sample = "BREAKING!!! Visit http://fake.site NOW - The government has 100% hidden the truth!"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
    print("Tokens   :", tokenize(clean_text(sample)))
    print("Final    :", preprocess_text(sample))

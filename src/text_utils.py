"""Shared text utilities for the Twitter sentiment project.

Used by the training notebook and by the Streamlit app, so that exactly the same
preprocessing is applied at training time and at prediction time.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LABELS = ["negative", "neutral", "positive"]

# Words that matter for sentiment and must NOT be removed as "stopwords".
KEEP_WORDS = {"no", "not", "nor", "never", "nothing", "cannot", "without",
              "very", "too", "more", "most", "least", "against"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")


def decontract(text):
    """Expand contractions so that negation survives as the token 'not'."""
    text = text.replace("\u2019", "'")
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    return text


def clean_text(text, remove_stopwords=False):
    """Heavy cleaning for bag-of-words models (TF-IDF + linear model)."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = decontract(text)
    text = REPEAT_RE.sub(r"\1\1", text)          # sooooo -> soo
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def light_clean(text):
    """Minimal cleaning for transformer models (they handle raw text well)."""
    text = URL_RE.sub("http", str(text))
    text = MENTION_RE.sub("@user", text)
    return " ".join(text.split())


def explain_linear(pipeline, cleaned_text, top_k=8):
    """Explain one prediction of a TF-IDF + linear-classifier sklearn Pipeline.

    Returns (classes, probabilities, predicted_label, contributions) where
    `contributions` is a list of (term, score) sorted by absolute size. A score is
    tfidf_value * coefficient for the predicted class: positive scores pushed the
    prediction towards that class, negative scores pushed against it.
    """
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    probs = pipeline.predict_proba([cleaned_text])[0]
    classes = list(clf.classes_)
    idx = int(np.argmax(probs))

    row = vec.transform([cleaned_text]).tocoo()
    names = vec.get_feature_names_out()
    coefs = clf.coef_[idx]
    contribs = [(names[j], float(v * coefs[j])) for j, v in zip(row.col, row.data)]
    contribs.sort(key=lambda kv: -abs(kv[1]))
    return classes, probs, classes[idx], contribs[:top_k]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:42:22 2021

@author: ivanr
"""
# pylint: disable=E1101
# =============================================================================
# Imports
# =============================================================================
import time
import pickle
from datetime import timedelta
from itertools import zip_longest
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import (
    MODEL_TYPE_SEMANTIC_SIMILARITY,
    BATCH_SIZE_SEMANTIC_SIMILARITY,
)

ENGLISH_STOPWORDS = set(stopwords.words("english"))
# nltk.download("wordnet")
LEMMATIZER = nltk.wordnet.WordNetLemmatizer()
SEMANTIC_SIMILARITY_MODEL = SentenceTransformer(MODEL_TYPE_SEMANTIC_SIMILARITY)

# =============================================================================
# Functions
# =============================================================================


def n_gram_novelty_calculator(summary: str, body: str) -> tuple:
    """
    Calculate novelty for n-grams 1 to 3.

    Calculates summary novelty by comparing the percentage of new tokens
    in the summary that dont appear in the body.
    This is done using a lemmatizer so that grammatical inflections do not
    affect the text.
    Furthermore, stopwords are removed.

    Parameters
    ----------
    summary : str
    body : str

    Returns
    -------
    float
        As fractions  (novelty_tokens, novelty_bigrams, novelty_trigrams).

    """
    summary = nltk.word_tokenize(summary)
    body = nltk.word_tokenize(body)

    summary = [
        LEMMATIZER.lemmatize(word) for word in summary if word not in ENGLISH_STOPWORDS
    ]
    body = [
        LEMMATIZER.lemmatize(word) for word in body if word not in ENGLISH_STOPWORDS
    ]

    # Generate sets of bigrams, trigrams and tokens for body and summary
    tokens_summary = set(summary)
    tokens_body = set(body)
    bigrams_summary = set(ngrams(summary, 2))
    bigrams_body = set(ngrams(body, 2))
    trigrams_summary = set(ngrams(summary, 3))
    trigrams_body = set(ngrams(body, 3))

    # Calculate novelty
    novel_tokens = tokens_summary - tokens_body
    novelty_tokens = len(novel_tokens) / len(tokens_summary)
    novel_bigrams = bigrams_summary - bigrams_body
    novelty_bigrams = len(novel_bigrams) / len(bigrams_summary)
    novel_trigrams = trigrams_summary - trigrams_body
    novelty_trigrams = len(novel_trigrams) / len(trigrams_summary)

    return novelty_tokens, novelty_bigrams, novelty_trigrams


# =============================================================================
# Semantic
# =============================================================================


def compute_semantic_similarity(batch):
    """Computes semantic similarity between summary and body."""

    # Retrieve (id, summary, text)
    to_encode = []
    id_list = []
    for row in batch:
        id_list.append(row[0])
        summary = row[2]
        body = "".join(pickle.loads(row[3]))
        to_encode += [summary]
        to_encode += [body]

    # Embbed all texts
    all_embeddings = SEMANTIC_SIMILARITY_MODEL.encode(to_encode, batch_size=100)
    query_embeddings, passage_embeddings = all_embeddings[::2], all_embeddings[1::2]

    all_similarity = []
    for idx, (q_emb, p_emb) in enumerate(zip(query_embeddings, passage_embeddings)):
        semantic_similarity = util.dot_score(q_emb, p_emb).numpy()[0][0]
        pageid = id_list[idx]
        all_similarity.append(
            {"pageid": pageid, "semantic_similarity": semantic_similarity}
        )
    return all_similarity


# start = time.time()
# end = time.time()
# str(timedelta(seconds=end - start))

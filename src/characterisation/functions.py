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
from datetime import timedelta
from itertools import zip_longest
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import MODEL_TYPE_SEMANTIC_SIMILARITY

ENGLISH_STOPWORDS = set(stopwords.words("english"))
# nltk.download("wordnet")
LEMMATIZER = nltk.wordnet.WordNetLemmatizer()
SEMANTIC_SIMILARITY_MODEL = SentenceTransformer(MODEL_TYPE_SEMANTIC_SIMILARITY)

# =============================================================================
# Functions
# =============================================================================


def novelty_calculator(summary: str, body: str) -> float:
    """
    Calculate novelty, speed: 67.9 it/s .

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
        novelty as a fraction.

    """
    summary = set(nltk.word_tokenize(summary)) - ENGLISH_STOPWORDS
    body = set(nltk.word_tokenize(body)) - ENGLISH_STOPWORDS

    summary = {LEMMATIZER.lemmatize(word) for word in summary}
    body = {LEMMATIZER.lemmatize(word) for word in body}

    novel_tokens = summary - body
    novelty = len(novel_tokens) / len(summary)
    return novelty


def n_gram_novelty_calculator(summary: str, body: str) -> tuple:
    """
    Calculate novelty for n-grams 1 to 3, speed: 57.28 it/s .

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
# import pickle
# from tqdm import tqdm
# from src.data.wikipedia.wiki_data_base import retrive_suitable_strings

# data = retrive_suitable_strings(limit=100)

# # id, summary, text
# data_semantic = [[i[0], i[2], pickle.loads(i[3])] for i in data]


# to_encode = []

# for row in tqdm(data_semantic):
#     summary = row[1]
#     body = row[2]
#     # body_sentences = list(itertools.chain(*[tokenize_sentence(i) for i in body]))
#     body_sentences = "".join(body)
#     to_encode += [summary]
#     to_encode += [body_sentences]

#     # body_sentences = "".join(body)


# start = time.time()
# all_embeddings = SEMANTIC_SIMILARITY_MODEL.encode(to_encode, batch_size=100)

# end = time.time()
# str(timedelta(seconds=end - start))


# query_embeddings, passage_embeddings = all_embeddings[::2], all_embeddings[1::2]


# all_similarity = []
# for idx, (q_emb, p_emb) in enumerate(zip(query_embeddings, passage_embeddings)):
#     print(q_emb)
#     similarity = util.dot_score(q_emb, p_emb).numpy()[0][0]
#     print(similarity)
#     row = data_semantic[idx]
#     summary = row[1]
#     body_sentences = "".join(row[2])

#     all_similarity.append([similarity, row[0], summary, body_sentences])

# import pandas as pd


# evaluate = [[score, sent] for score, sent in zip(similarity, body_sentences)]
# import pandas as pd


# row_to_look = data_semantic[73]
# sum_to_look = row_to_look[1]
# body_to_look = "".join(row_to_look[2])

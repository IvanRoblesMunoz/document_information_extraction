#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:46:20 2021

@author: ivanr
"""
# =============================================================================
# Imports
# =============================================================================
import pickle
import pandas as pd

from src.data.data_statics import (
    MIN_SEMANTIC_SIMILARITY,
    MIN_NOVELTY,
    MAX_NOVELTY,
    MAX_TOKENS_BODY,
)
from src.data.wikipedia.wiki_data_base import (
    retrieve_query,
    retrive_observations_from_ids,
)


def get_article_ids(
    min_semantic_similarity=MIN_SEMANTIC_SIMILARITY,
    max_novelty=MAX_NOVELTY,
    min_novelty=MIN_NOVELTY,
    max_tokens_body=MAX_TOKENS_BODY,
    random_state=None,
    n_sample_texts=None,
):
    query_suitable_articles = f"""
    SELECT ar.*,
           nv.novelty_tokens,
           nv.novelty_bigrams,
           nv.novelty_trigrams,
           cs.semantic_similarity
           
    FROM article_level_info ar
    INNER JOIN wiki_article_novelty nv
        ON ar.pageid = nv.pageid
    INNER JOIN wiki_article_cosine_similarity cs
        ON ar.pageid = cs.pageid
    WHERE cs.semantic_similarity>={min_semantic_similarity}
        AND nv.novelty_tokens<={max_novelty}
        AND nv.novelty_tokens>={min_novelty}
        AND ar.body_word_count<={max_tokens_body}
    """

    characterisation_df = pd.DataFrame(
        retrieve_query(query_suitable_articles),
        columns=[
            "pageid",
            "title",
            "summary_word_count",
            "body_word_count",
            "novelty_tokens",
            "novelty_bigrams",
            "novelty_trigrams",
            "semantic_similarity",
        ],
    )

    if not n_sample_texts:
        n_sample_texts = len(characterisation_df)

    pageids_to_evaluate = list(
        characterisation_df["pageid"].sample(
            n=n_sample_texts, random_state=random_state
        )
    )
    return pageids_to_evaluate

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:51:54 2021

@author: roblesi
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pickle

from src.data.wikipedia.wiki_data_base import retrieve_query
from src.retriever.run_make_faiss_indeces import (
    create_faiss_document_store,
    initialise_faiss_retriever,
    embed_article_batch,
)

# =============================================================================
# Statics
# =============================================================================
from src.retriever.retriever_statics import FAISS_TEMP_SQL_DB_PATH


TESTING_QUERY = """
SELECT *
FROM faiss_embedding_store

LIMIT 1000
"""

# =============================================================================
# Tests
# =============================================================================
# Instantiate embedding objects
document_store = create_faiss_document_store()
retriever = initialise_faiss_retriever(document_store)

article_batch = retrieve_query(TESTING_QUERY, out_f=FAISS_TEMP_SQL_DB_PATH)


# TODO: correct pageid


def decode_article(article):
    pageid = article[0]
    title = article[1]
    embeddings = pickle.loads(article[2])
    body_sections = pickle.loads(article[3])
    return (pageid, title, embeddings, body_sections)


def test_embeddings_are_consistent():
    for article in article_batch:
        pageid, title, embeddings, body_sections = decode_article(article)

        for i in range(len(body_sections)):
            print(i)

            test_embedding = embed_article_batch(retriever, [body_sections[i]])
            candidate_embedding = np.reshape(embeddings[i], test_embedding.shape)

            error1 = (
                abs(test_embedding - candidate_embedding).mean()
                / abs(test_embedding).mean()
                * 100
            )
            error2 = (
                abs(test_embedding - candidate_embedding).mean()
                / abs(candidate_embedding).mean()
                * 100
            )

            assert error1, error2 < 1e-2


test_embeddings_are_consistent()

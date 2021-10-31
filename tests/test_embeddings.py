#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:51:54 2021

@author: roblesi
"""
# =============================================================================
# Imports
# =============================================================================
import sys
import os
import pytest
import pickle
from pathlib import Path

import numpy as np

from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from src.data.wikipedia.wiki_data_base import retrieve_query
from src.retriever.run_make_faiss_indeces import (
    create_faiss_document_store,
    initialise_faiss_retriever,
    embed_article_batch,
)


# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import SQL_WIKI_DUMP
from src.retriever.retriever_statics import (
    FAISS_TEMP_SQL_DB_PATH,
    FAISS_DB_DATA_PATH,
    FAISS_QUERY_EMBEDDING_MODEL,
    FAISS_PASSAGE_EMBEDDING_MODEL,
)


TESTING_PAGE_IDS = [
    145422,
    25817778,
    35708276,
    236034,
    21377251,
    2150841,
    12153654,
    46230181,
    15910,
    34228206,
]

TESTING_QUERY_EMBEDDINGS = f"""
SELECT *
FROM faiss_embedding_store
WHERE pageid in ({",".join([str(i) for i in TESTING_PAGE_IDS])})
"""

TESTING_QUERY_ARTICLES = f"""
SELECT wk.pageid,
       wk.summary,
       wk.body_sections,
       ar.title
FROM wiki_articles wk
LEFT JOIN article_level_info ar
    ON wk.pageid=ar.pageid
    
WHERE wk.pageid in ({",".join([str(i) for i in TESTING_PAGE_IDS])})
"""

# =============================================================================
# Utility functions
# =============================================================================
def decode_article_embeddings(article):
    pageid = article[0]
    title = article[1]
    embeddings = pickle.loads(article[2])
    body_sections = pickle.loads(article[3])
    return (pageid, title, embeddings, body_sections)


def decode_article_text(article):
    pageid = article[0]
    summary = article[1]
    body_sections = pickle.loads(article[2])
    title = article[3]

    text = summary + "".join(body_sections)
    return pageid, text, title


def initialise_faiss_retriever(
    document_store: FAISSDocumentStore,
) -> DensePassageRetriever:
    """Initialise passage retriever."""
    return DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=FAISS_QUERY_EMBEDDING_MODEL,
        passage_embedding_model=FAISS_PASSAGE_EMBEDDING_MODEL,
        use_gpu=True,
        embed_title=True,
    )


def create_faiss_document_store() -> FAISSDocumentStore:
    """Create new document store."""
    return FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        sql_url=FAISS_DB_DATA_PATH,
        return_embedding=True,
    )


def embed_article_batch(retriever, flattend_article_batch):
    return retriever.embed_passages(flattend_article_batch)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="class")
def make_retriever():
    # Instantiate embedding objects
    document_store = create_faiss_document_store()
    retriever = initialise_faiss_retriever(document_store)
    return retriever


@pytest.fixture(scope="class")
def make_article_embbedding_batch():
    article_batch = retrieve_query(
        TESTING_QUERY_EMBEDDINGS, out_f=FAISS_TEMP_SQL_DB_PATH
    )
    article_batch = [decode_article_embeddings(article) for article in article_batch]
    article_batch = {article[0]: article[1:] for article in article_batch}
    return article_batch


@pytest.fixture(scope="class")
def make_article_body_batch():
    article_batch = retrieve_query(TESTING_QUERY_ARTICLES, out_f=SQL_WIKI_DUMP)
    article_batch = [decode_article_text(article) for article in article_batch]
    article_batch = {article[0]: article[1:] for article in article_batch}
    return article_batch


@pytest.mark.usefixtures(
    "make_retriever", "make_article_embbedding_batch", "make_article_body_batch"
)
class TestFaissEmbeddingsDB:
    @pytest.mark.parametrize("pageid", [(i) for i in TESTING_PAGE_IDS])
    def test_embeddings_are_consistent(
        self, pageid, make_retriever, make_article_embbedding_batch
    ):

        title, embeddings, body_sections = make_article_embbedding_batch[pageid]
        retriever = make_retriever

        for i in range(len(body_sections)):

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

    @pytest.mark.parametrize("pageid", [(i) for i in TESTING_PAGE_IDS])
    def test_articles_and_page_ids_are_consistent(
        self,
        pageid,
        make_article_embbedding_batch,
        make_article_body_batch,
    ):
        title_emb, embeddings_emb, body_sections_emb = make_article_embbedding_batch[
            pageid
        ]

        text_txt, title_txt = make_article_body_batch[pageid]

        for section in body_sections_emb:
            assert title_emb == section.meta["title"]

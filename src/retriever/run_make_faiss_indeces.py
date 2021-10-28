#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:29:04 2021

@author: roblesi
"""

# =============================================================================
# Imports
# =============================================================================
import os
import sys
import itertools

# import multiprocessing
from pathlib import Path
import pickle


WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever

from haystack.preprocessor import PreProcessor
from src.retriever.retriever_input_articles import processed_document_generator
from src.data.wikipedia.wiki_data_base import get_connection, FAISSEmbeddingStore

# =============================================================================
# Statics
# =============================================================================
# from src.data.data_statics import SQL_WIKI_DUMP # TODO: Change this
from src.retriever.retriever_statics import (
    FAISS_MAX_PASSAGE_TOKEN_LEN,
    FAISS_DB_DATA_PATH,
    FAISS_QUERY_EMBEDDING_MODEL,
    FAISS_PASSAGE_EMBEDDING_MODEL,
    FAISS_TEMP_SQL_DB_PATH,
    # FAISS_EMB_DATA_PATH,
    # FAISS_CONFIG_DATA_PATH,
)

# SQL_WIKI_DUMP = "/home/roblesi/git/document_information_extraction/data/interim/wiki_db_dumps_for_faiss.db"

# =============================================================================
# FAISS document store
# =============================================================================


def create_faiss_document_store() -> FAISSDocumentStore:
    """Create new document store."""
    return FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        sql_url=FAISS_DB_DATA_PATH,
        return_embedding=True,
    )


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


def embed_article_batch(retriever, flattend_article_batch):
    return retriever.embed_passages(flattend_article_batch)


def aggregate_output_to_store_by_page_id(flattend_article_batch, flattended_emb_batch):
    prev_page_id = None
    entry_docs = []
    entry_embs = []
    batch_entry = []
    for passage, emb in zip(flattend_article_batch, flattended_emb_batch):

        curr_page_id = passage.meta["page_id"]

        if (curr_page_id == prev_page_id) | (prev_page_id is None):
            entry_docs.append(passage)
            entry_embs.append(emb)
            prev_page_id = curr_page_id
        else:
            encoded_docs = pickle.dumps(entry_docs)
            encoded_embs = pickle.dumps(entry_embs)
            title = entry_docs[0].meta["title"]
            print(prev_page_id)  # TODO: remove this
            batch_entry.append(
                {
                    "pageid": prev_page_id,
                    "title": title,
                    "embeddings": encoded_embs,
                    "body_sections": encoded_docs,
                }
            )
            entry_docs = []
            entry_embs = []
            prev_page_id = curr_page_id

            entry_docs.append(passage)
            entry_embs.append(emb)
    # Do the same for last page id
    encoded_docs = pickle.dumps(entry_docs)
    encoded_embs = pickle.dumps(entry_embs)
    title = entry_docs[0].meta["title"]
    print(prev_page_id)  # TODO: remove this
    batch_entry.append(
        {
            "pageid": prev_page_id,
            "title": title,
            "embeddings": encoded_embs,
            "body_sections": encoded_docs,
        }
    )
    entry_docs = []
    entry_embs = []
    prev_page_id = curr_page_id

    entry_docs.append(passage)
    entry_embs.append(emb)

    return batch_entry


def insert_embeddings_to_database(
    batch_entry, table=FAISSEmbeddingStore, out_f=FAISS_TEMP_SQL_DB_PATH
):
    engine, session = get_connection(out_f=out_f)
    engine.execute(table.__table__.insert(), batch_entry)
    session.commit()
    session.close()


def main():
    # Instantiate  embedding objects
    document_store = create_faiss_document_store()
    retriever = initialise_faiss_retriever(document_store)

    # Instantiate generator objects
    faiss_preprocessor = PreProcessor(split_length=FAISS_MAX_PASSAGE_TOKEN_LEN)
    article_generator_faiss = processed_document_generator(
        storage_method="faiss",
        preprocessor=faiss_preprocessor,
        batch_size_doc_generator=1000,
        # **{"n_sample_articles": 100}
    )

    for article_batch in article_generator_faiss:
        flattend_article_batch = list(itertools.chain(*article_batch))

        # This should be its own process
        flattended_emb_batch = embed_article_batch(retriever, flattend_article_batch)

        batch_entry = aggregate_output_to_store_by_page_id(
            flattend_article_batch, flattended_emb_batch
        )

        insert_embeddings_to_database(batch_entry)


if __name__ == "__main__":
    main()

# 2e6 / 7308528
7308528 * 0.575

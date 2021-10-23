#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 19:47:05 2021

@author: roblesi
"""
# =============================================================================
# Imports
# =============================================================================
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever

from haystack.preprocessor import PreProcessor
from src.retriever.retriever_input_articles import processed_document_generator

# =============================================================================
# Statics
# =============================================================================
from src.retriever.retriever_statics import (
    FAISS_MAX_PASSAGE_TOKEN_LEN,
    FAIIS_DB_DATA_PATH,
    FAIIS_EMB_DATA_PATH,
    FAIIS_QUERY_EMBEDDING_MODEL,
    FAIIS_PASSAGE_EMBEDDING_MODEL,
)

# =============================================================================
# FAIIS document store
# =============================================================================


def create_faiss_document_store() -> FAISSDocumentStore:
    """Create new document store."""
    return FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        sql_url=FAIIS_DB_DATA_PATH,
        return_embedding=True,
    )


def initialise_faiis_retriever(
    document_store: FAISSDocumentStore,
) -> DensePassageRetriever:
    """Initialise passage retriever."""
    return DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=FAIIS_QUERY_EMBEDDING_MODEL,
        passage_embedding_model=FAIIS_PASSAGE_EMBEDDING_MODEL,
        use_gpu=True,
        embed_title=True,
    )


def load_existing_faiss_document_store() -> FAISSDocumentStore:
    return FAISSDocumentStore.load(
        index_path=str(FAIIS_EMB_DATA_PATH),
    )


def save_faiss_document_store(document_store: FAISSDocumentStore) -> None:
    document_store.save(FAIIS_EMB_DATA_PATH)


document_store = create_faiss_document_store()
document_store.delete_documents()

retriever = initialise_faiis_retriever(document_store)


def populate_faiis_document_store(document_store, retriever):
    """
    # full_time (1000 articles)
        78.95658421516418
    # time_in_encode
        49.14044761657715
    # time_in_assign_embed
        0.005559444427490234
    # time_in_write
        26.85038423538208
    # remainder (Mainly read and preprocess)
        2.960192918777466
    """

    faiss_pre_processor = PreProcessor(
        split_length=FAISS_MAX_PASSAGE_TOKEN_LEN,
    )

    faiss_generator = processed_document_generator(
        storage_method="faiss",
        preprocessor=faiss_pre_processor,
        **{"n_sample_articles": None}
    )

    for doc_batch in faiss_generator:

        batch_emb = retriever.embed_passages(doc_batch)

        for emb, doc in zip(batch_emb, doc_batch):
            doc.embedding = emb

        document_store.write_documents(doc_batch)

        document_store.save(FAIIS_EMB_DATA_PATH)

    # document_store.update_embeddings(retriever=retriever)


populate_faiis_document_store(document_store, retriever)

# doc_batch[0]
# answers = retriever.retrieve("What are the cause of Autism?")
# a = answers[0].__dict__
# doc_batch[0]

# [{"content": answer.content, "title": answer.meta} for answer in answers]

# help(PreProcessor)
# document_store.delete_documents()
# document_store.write_documents(docs)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:52:41 2021

@author: roblesi
"""
# =============================================================================
# Paths
# =============================================================================
import os
from pathlib import Path

WORKING_DIRECTORY = Path(os.getcwd())
DATA_PATH = WORKING_DIRECTORY / "data"
INTERIM_DATA_PATH = DATA_PATH / "interim"

# =============================================================================
# Dense Passage Retriever statics
# =============================================================================
FAISS_MAX_PASSAGE_TOKEN_LEN = 150  # 100
FAISS_GEN_N_ARTICLES_BATCH = 1000

FAISS_DB_DATA_PATH = "sqlite:///" + str(INTERIM_DATA_PATH / "faiss_sql_database.db")
FAISS_TEMP_SQL_DB_PATH = INTERIM_DATA_PATH / "temp_faiss_document_store.db"
FAISS_EMB_DATA_PATH = INTERIM_DATA_PATH / "faiss_sql_database.faiss"
FAISS_CONFIG_DATA_PATH = INTERIM_DATA_PATH / "faiss_sql_database.json"

FAISS_QUERY_EMBEDDING_MODEL = "facebook/dpr-question_encoder-single-nq-base"
FAISS_PASSAGE_EMBEDDING_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"
FAISS_N_ARTICLES_ENCODE = 2_200_000
FAISS_ARTICLES_BATCH_SIZE = 1000

# =============================================================================
# Sparse retriever statics
# =============================================================================
BM25_MAX_PASSAGE_TOKEN_LEN = 2500
BM25_TEMP_SQL_DB_PATH = INTERIM_DATA_PATH / "temp_bm25_document_store.db"

# =============================================================================
# ODQA statics
# =============================================================================
BM25_PASSAGE_WEIGHT = 3
BM25_TITLE_WEIGHT = 1
BM25_LIMIT_SEARCH = 1000

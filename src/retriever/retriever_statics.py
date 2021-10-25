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
FAISS_MAX_PASSAGE_TOKEN_LEN = 250  # 200
FAISS_GEN_N_ARTICLES_BATCH = 1000

FAISS_DB_DATA_PATH = "sqlite:///" + str(INTERIM_DATA_PATH / "faiss_sql_database.db")
FAISS_TEMP_SQL_DB_PATH = INTERIM_DATA_PATH / "temp_faiss_document_store.db"
FAISS_EMB_DATA_PATH = INTERIM_DATA_PATH / "faiss_sql_database.faiss"
FAISS_CONFIG_DATA_PATH = INTERIM_DATA_PATH / "faiss_sql_database.json"

FAISS_QUERY_EMBEDDING_MODEL = "facebook/dpr-question_encoder-single-nq-base"
FAISS_PASSAGE_EMBEDDING_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"

# =============================================================================
# Sparse retriever statics
# =============================================================================
ELASTICSEARCH_MAX_PASSAGE_TOKEN_LEN = 2500

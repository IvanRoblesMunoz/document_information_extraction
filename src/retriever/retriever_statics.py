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
FAISS_MAX_PASSAGE_TOKEN_LEN = 200
FAIIS_DB_DATA_PATH = "sqlite:///" + str(INTERIM_DATA_PATH / "faiis_sql_database.db")
FAIIS_EMB_DATA_PATH = INTERIM_DATA_PATH / "faiis_sql_database.faiss"
FAIIS_QUERY_EMBEDDING_MODEL = "facebook/dpr-question_encoder-single-nq-base"
FAIIS_PASSAGE_EMBEDDING_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"

# =============================================================================
# Sparse retriever statics
# =============================================================================
ELASTICSEARCH_MAX_PASSAGE_TOKEN_LEN = 2500

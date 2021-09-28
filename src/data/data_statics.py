#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:06:40 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
WORKING_DIRECTORY = Path(os.getcwd())
DATA_PATH = WORKING_DIRECTORY / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
RAW_WIKIPEDIA_CORPUS = RAW_DATA_PATH / "enwiki-latest-pages-articles.xml.bz2"

# =============================================================================
# Wikipedia parsing constants
# =============================================================================
SQL_WIKI_DUMP = INTERIM_DATA_PATH / "wiki_db_dumps.db"
READ_QUE_SIZE = 1
SQL_QUE_SIZE = 1
N_PROCESSES = os.cpu_count()
BATCH_SIZE = 100  # 00

# =============================================================================
# Summary suitability constants
# =============================================================================
MIN_TOKENS_SUMMARY = 40
MAX_TOKENS_SUMMARY = 512
MIN_TOKENS_BODY = 250
MIN_COMPRESION_RATIO = 0.05
MAX_COMPRESION_RATIO = 0.3

# =============================================================================
# Dataset characterisation constants
# =============================================================================
# We will use this model, since it is the second highest rank and it accepts
# a larger input size than the highest ranked ()
MAX_MODEL_SQUENCE_LENGTH = 512
MODEL_TYPE_SEMANTIC_SIMILARITY = "all-mpnet-base-v1"
BATCH_SIZE_SEMANTIC_SIMILARITY = 64
SEM_SIM_READ_QUE_SIZE = 1
SEM_SIM_PREP_QUE_SIZE = 1
SEM_SIM_SQL_QUE_SIZE = 1
SEM_SIM_N_PROCESSES = 4  # os.cpu_count() - 2


BATCH_SIZE_NOVELTY = 1000
NOVELTY_READ_QUE_SIZE = 1
NOVELTY_SQL_QUE_SIZE = 1
NOVELTY_N_PROCESSES = os.cpu_count()

TEMP_DB = INTERIM_DATA_PATH / "temp_database.db"

# --- Bertscore statics ---
# We will use the highest performing model that has an input length long enough
# for our task.
# Perfromance link: https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?usp=sharing
# Berstcore link: https://github.com/Tiiiger/bert_score

BERTSCORE_MODEL = "allenai/led-base-16384"
BERTSCORE_MODEL_LAYER = 6
BERTSCORE_LANGUAGE = "en"
BERTSCORE_ALL_LAYERS = False

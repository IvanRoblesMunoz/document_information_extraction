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
DATA_PATH = Path("/home/ivanr/git/document_information_extraction/data/")
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
RAW_WIKIPEDIA_CORPUS = RAW_DATA_PATH / "enwiki-latest-pages-articles.xml.bz2"
DECOMPRESSED_WIKIPEDIA_DUMP = RAW_DATA_PATH / "enwiki-latest-pages-articles.xml"
SQL_WIKI_DUMP = INTERIM_DATA_PATH / "wiki_db_dumps.db"
# =============================================================================
# Wikipedia parsing constants
# =============================================================================
READ_QUE_SIZE = 1  # os.cpu_count()
SQL_QUE_SIZE = 1  # os.cpu_count()
N_PROCESSES = os.cpu_count()
BATCH_SIZE = 10000

# =============================================================================
# Wiki SQL dump constants
# =============================================================================
MIN_TOKENS_SUMMARY = 40
MIN_TOKENS_BODY = 250
MIN_COMPRESION_RATIO = 0.05
MAX_COMPRESION_RATIO = 0.4

# =============================================================================
# Bert score constants
# =============================================================================
MODEL_TYPE = "allenai/led-base-16384"
NUM_LAYERS = 6
ALL_LAYERS = False
LANGUAGE = "en"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:06:40 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
DATA_PATH = Path("/home/ivanr/git/document_information_extraction/data/")
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
RAW_WIKIPEDIA_CORPUS = RAW_DATA_PATH / "enwiki-latest-pages-articles.xml.bz2"
DECOMPRESSED_WIKIPEDIA_DUMP = RAW_DATA_PATH / "enwiki-latest-pages-articles.xml"
SQL_WIKI_DUMP = INTERIM_DATA_PATH / "wiki_db_dumps_test.db"

# =============================================================================
# Wiki SQL dump constants
# =============================================================================
MIN_TOKENS_SUMMARY = 100
MIN_SUMMARY_RATIO = 0.05
MAX_SUMMARY_RATIO = 0.3
MAX_TOKENS_BODY = 16384
# =============================================================================
# Bert score constants
# =============================================================================
MODEL_TYPE = "allenai/led-base-16384"
NUM_LAYERS = 6
ALL_LAYERS = False
LANGUAGE = "en"

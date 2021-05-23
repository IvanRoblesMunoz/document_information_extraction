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
# Constants and paths
# =============================================================================
DATA_PATH = Path("/home/ivanr/git/document_information_extraction/data/")
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
RAW_WIKIPEDIA_CORPUS = RAW_DATA_PATH / "enwiki-latest-pages-articles.xml.bz2"
SQL_WIKI_DUMP = INTERIM_DATA_PATH / "wiki_db_dumps.db"

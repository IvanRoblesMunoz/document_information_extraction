#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 10:11:08 2021

@author: roblesi
"""

# =============================================================================
# Imports
# =============================================================================
import os
import sys
from pathlib import Path
import pickle
from tqdm import tqdm
from haystack.preprocessor import PreProcessor

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from src.retriever.retriever_input_articles import processed_document_generator
from src.retriever.database_temp_bm25 import (
    insert_into_bm25_table,
    connect_to_bm25_temp_db,
    create_bm25_table,
)
from src.data.wikipedia.wiki_data_base import retrieve_query_in_batches

# =============================================================================
# Statics
# =============================================================================
from src.retriever.retriever_statics import (
    BM25_MAX_PASSAGE_TOKEN_LEN,
    BM25_TEMP_SQL_DB_PATH,
)
from src.data.data_statics import SQL_WIKI_DUMP

print(SQL_WIKI_DUMP)
# =============================================================================
# Functions
# =============================================================================

bm25_preprocessor = PreProcessor(split_length=BM25_MAX_PASSAGE_TOKEN_LEN)
article_generator = processed_document_generator(
    storage_method="bm25",
    preprocessor=bm25_preprocessor,
    batch_size_doc_generator=10_000,
)

if os.path.isfile(BM25_TEMP_SQL_DB_PATH):
    os.remove(BM25_TEMP_SQL_DB_PATH)


def format_article_batch_to_insert(article_bach):
    """Format articles to insert into main database table."""
    formated_article_batch = []
    for article in article_bach:
        for passage_dict in article:
            passage = passage_dict["content"]
            pageid = passage_dict["meta"]["page_id"]
            title = passage_dict["meta"]["title"]

            formated_article_batch.append((pageid, passage, title))
    return formated_article_batch


def main():
    """Run make BM25 indeces."""
    # Create table and produce cursors
    db, cur = connect_to_bm25_temp_db()
    create_bm25_table(cur, db)

    for article_bach in article_generator:
        formated_article_batch = format_article_batch_to_insert(article_bach)
        insert_into_bm25_table(formated_article_batch, cur, db)


if __name__ == "__main__":
    # main()

    db, cur = connect_to_bm25_temp_db(out_f=str(SQL_WIKI_DUMP))

    create_bm25_table(cur, db)

    query_transfer = """
    SELECT *
    FROM bm25_wiki_articles
    """

    transfer_generator = retrieve_query_in_batches(
        query_transfer, out_f=BM25_TEMP_SQL_DB_PATH
    )

    for batch in tqdm(transfer_generator):
        insert_into_bm25_table(batch, cur, db)

    # Delete temporary db
    if os.path.exists(BM25_TEMP_SQL_DB_PATH):
        os.remove(BM25_TEMP_SQL_DB_PATH)

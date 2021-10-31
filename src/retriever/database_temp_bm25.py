#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 09:36:36 2021

@author: roblesi
"""
# =============================================================================
# Imports
# =============================================================================
import sqlite3

from src.data.wikipedia.wiki_data_base import retrieve_query

# =============================================================================
# Statics
# =============================================================================
from src.retriever.retriever_statics import BM25_TEMP_SQL_DB_PATH


def connect_to_bm25_temp_db(out_f=BM25_TEMP_SQL_DB_PATH):
    db = sqlite3.connect(out_f)
    cur = db.cursor()
    return db, cur


def create_bm25_table(cur, db):
    cur.execute(
        """
        CREATE VIRTUAL TABLE bm25_wiki_articles
        USING FTS5(pageid
                   ,passage
                   ,title
                   ,tokenize="porter unicode61")
        """
    )
    db.commit()


def insert_into_bm25_table(article_batch, cur, db):
    cur.executemany(
        "insert into bm25_wiki_articles (pageid, passage, title) values (?,?,?);",
        article_batch,
    )
    db.commit()

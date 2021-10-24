#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:05:11 2021

@author: roblesi
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sqlite3

# =============================================================================
# Statics
# =============================================================================

from src.data.wikipedia.wiki_data_base import get_connection, ArticlesInFAISS
from src.retriever.retriever_statics import FAISS_TEMP_SQL_DB_PATH

# =============================================================================
# Functions
# =============================================================================


def engine_to_faiss_temp_db():
    """Start engine and session to FAISS temp database."""
    engine, session = get_connection(out_f=FAISS_TEMP_SQL_DB_PATH)
    return engine, session


def retrieve_tables_in_database(conn):
    """Retrieve table names."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return tables


def connect_to_faiss_temp_db():
    """Connects to faiss db."""
    conn = sqlite3.connect(FAISS_TEMP_SQL_DB_PATH)

    # Create tables if they dont exist yet
    tables = retrieve_tables_in_database(conn)
    if len(tables) == 0:
        _ = engine_to_faiss_temp_db()
    return conn


def retrieve_article_page_ids_already_in_faiss(conn):
    """Retrieve page ids that have already being inserted."""
    cursor = conn.cursor()

    query = """
            SELECT DISTINCT pageid
            FROM faiss_embedding_store
            """
    cursor.execute(query)
    existing_ids = cursor.fetchall()
    return existing_ids


def insert_existing_faiss_articles() -> None:
    """Insert page ids to table to indicate they have already been inserted"""
    print("reading inserted page_ids...")
    # Connect to FAISS db and retrieve all page ids already inserted
    conn_faiss = connect_to_faiss_temp_db()
    existing_ids = retrieve_article_page_ids_already_in_faiss(conn_faiss)
    existing_ids = [{"pageid": int(i[0])} for i in existing_ids]
    print(f"{len(existing_ids)} articles already inserted")

    # Connect to wiki db
    engine_wk_db, session_wk_db = engine_to_faiss_temp_db()
    conn_wk_db = engine_wk_db.connect()

    # Refresh table by deleting it
    conn_wk_db.execute("DELETE FROM articles_in_faiss;")

    # Insert all pageids
    engine_wk_db.execute(ArticlesInFAISS.__table__.insert(), existing_ids)
    session_wk_db.commit()
    session_wk_db.close()

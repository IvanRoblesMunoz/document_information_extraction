#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:09:50 2021

@author: roblesi
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
from pathlib import Path
import pickle
import re
import time
from datetime import timedelta

from tqdm import tqdm

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from src.data.wikipedia.wiki_data_base import (
    retrieve_query_in_batches,
    get_connection,
    retrieve_query,
    ArticleRedirectFlag,
    transfer_to_new_db,
    redirect_flag_data_input_formater,
    count_articles,
)

# =============================================================================
# Statics
# =============================================================================

from src.data.data_statics import REDIRECT_INSERT_BUFFER_SIZE, TEMP_DB, SQL_WIKI_DUMP

REDIRECT_RE = re.compile("REDIRECT")

REDIRECT_TITLES = re.compile(
    "Category:|Wikipedia:|File:|Template:|Draft:|Portal:|Module:|Help:|MediaWiki:|Category :|Wikipedia :|File :|Template :|Draft :|Portal :|Module :|Help :|MediaWiki :"
)

# =============================================================================
# Functions
# =============================================================================


def decode_row(row: list) -> tuple:
    """Decode row from query."""
    page_id = row[0]
    summary = row[1]
    body_sections = pickle.loads(row[2])
    title = row[3]
    return page_id, summary, body_sections, title


def check_is_redirect(summary, title):
    """Check if article is a redirect."""
    contains_redirect = bool(re.search(REDIRECT_RE, summary))
    contains_redirect_title = bool(re.search(REDIRECT_TITLES, title))
    return contains_redirect | contains_redirect_title


def insert_articles_to_db(observations_to_insert, engine, session):
    """Insert article redirect flag into database."""
    engine.execute(ArticleRedirectFlag.__table__.insert(), observations_to_insert)
    session.commit()
    session.close()


def make_query_generator(buffer_size=REDIRECT_INSERT_BUFFER_SIZE):
    """Make flag indicating if the article is a redirect."""

    # Drop entries before restarting
    engine, session = get_connection()
    conn = engine.connect()
    conn.execute("DELETE FROM article_redirect_flag;")

    # Connect to temporary database
    engine_temp_db, session_temp_db = get_connection(TEMP_DB)
    conn_temp_db = engine_temp_db.connect()
    conn_temp_db.execute("DELETE FROM article_redirect_flag;")
    session_temp_db.commit()
    session_temp_db.close()

    # Make query to extract articles
    query = """
    SELECT wk.pageid,
           wk.summary, 
           wk.body_sections, 
           ar.title
    FROM wiki_articles wk
    LEFT JOIN article_level_info ar
        ON ar.pageid = wk.pageid

    """

    # count articles for progress par
    print("counting articles...")
    n_articles = count_articles(query)

    n_articles_processed = 0
    observations_to_insert = []
    # Iterate throug articles
    for article in tqdm(
        retrieve_query_in_batches(query, batchsize=1), total=n_articles
    ):
        # Make flag to insert
        page_id, summary, body_sections, title = decode_row(*article)
        redirect_flag = check_is_redirect(summary, title)
        obs = {"pageid": page_id, "redirect_flag": redirect_flag}
        observations_to_insert.append(obs)

        n_articles_processed += 1

        # Insert into db
        if n_articles_processed % buffer_size == 0:
            insert_articles_to_db(
                observations_to_insert, engine_temp_db, session_temp_db
            )
            observations_to_insert = []

    # Insert remaining articles
    if len(observations_to_insert) == 0:
        insert_articles_to_db(observations_to_insert, engine_temp_db, session_temp_db)


if __name__ == "__main__":
    start_time = time.time()
    make_query_generator(buffer_size=REDIRECT_INSERT_BUFFER_SIZE)

    # Transfer to main db
    src_query = """
        SELECT *
        FROM article_redirect_flag
    """
    transfer_to_new_db(
        src_query,
        src_db=TEMP_DB,
        dest_db=SQL_WIKI_DUMP,
        dest_table=ArticleRedirectFlag,
        batch_formater=redirect_flag_data_input_formater,
    )

    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    end_time = time.time()
    print("finished in: ", str(timedelta(seconds=end_time - start_time)))

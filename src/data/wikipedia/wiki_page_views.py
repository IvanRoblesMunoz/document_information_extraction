#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:29:13 2021

@author: roblesi
"""


# =============================================================================
# Imports
# =============================================================================
import os
import sys
import threading
import multiprocessing
from pathlib import Path
import datetime
import _thread as thread

from tqdm import tqdm

import numpy as np
from mwviews.api import PageviewsClient

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from src.data.wikipedia.wiki_data_base import (
    retrieve_query_in_batches,
    WikiPageView,
    get_connection,
    transfer_to_new_db,
    wiki_page_views_data_input_formater,
    retrieve_query,
)

# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import TEMP_DB, SQL_WIKI_DUMP

PAGE_VIEWS_START_DATE = "20201001"
PAGE_VIEWS_END_DATE = "20211001"
PAGE_VIEWS_API_TIMEOUT = 10
PAGE_VIEWS_REQUEST_BATCH = 10


# =============================================================================
# Functions
# =============================================================================


def quit_function(fn_name):
    """Quit finction."""
    print("{0} took too long".format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


def exit_after(seconds):
    """Decorator to exit process if function takes longer than s seconds."""

    def outer(func):
        def inner(*args, **kwargs):
            timer = threading.Timer(seconds, quit_function, args=[func.__name__])
            timer.start()
            try:
                result = func(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


@exit_after(PAGE_VIEWS_API_TIMEOUT)
def make_pageview_request(
    client_user: PageviewsClient,
    titles_flat: list,
) -> list:
    """Make request to download pageviews."""
    return client_user.article_views(
        "en.wikipedia",
        titles_flat,
        start=PAGE_VIEWS_START_DATE,
        end=PAGE_VIEWS_END_DATE,
        granularity="monthly",
    )


def format_response_into_entry(page_view_response_batch, page_id, title):
    """Format response to match correct entry for database."""
    try:
        row = {
            "pageid": page_id,
            "title": title,
            "pageviews_2020_10_01": page_view_response_batch[
                datetime.datetime(2020, 10, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2020_11_01": page_view_response_batch[
                datetime.datetime(2020, 11, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2020_12_01": page_view_response_batch[
                datetime.datetime(2020, 12, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_01_01": page_view_response_batch[
                datetime.datetime(2021, 1, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_02_01": page_view_response_batch[
                datetime.datetime(2021, 2, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_03_01": page_view_response_batch[
                datetime.datetime(2021, 3, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_04_01": page_view_response_batch[
                datetime.datetime(2021, 4, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_05_01": page_view_response_batch[
                datetime.datetime(2021, 5, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_06_01": page_view_response_batch[
                datetime.datetime(2021, 6, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_07_01": page_view_response_batch[
                datetime.datetime(2021, 7, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_08_01": page_view_response_batch[
                datetime.datetime(2021, 8, 1, 0, 0)
            ][title.replace(" ", "_")],
            "pageviews_2021_09_01": page_view_response_batch[
                datetime.datetime(2021, 9, 1, 0, 0)
            ][title.replace(" ", "_")],
        }
        relevant_vals = list(row.values())[2:]
        relevant_vals = [i if i else 0 for i in relevant_vals]

        row["mean_views"] = np.mean(relevant_vals)
    except KeyError:
        print(f"Key error {title}")

    return row


def insert_articles_to_db(observations_to_insert, engine, session):
    """Insert article redirect flag into database."""
    engine.execute(WikiPageView.__table__.insert(), observations_to_insert)
    session.commit()
    session.close()


def update_pageviews_already_downloaded():
    """Update pagevies from temp db into main db if already downloaded."""
    src_query = """
        SELECT *
        FROM wiki_page_view
    """
    transfer_to_new_db(
        src_query,
        src_db=TEMP_DB,
        dest_db=SQL_WIKI_DUMP,
        dest_table=WikiPageView,
        batch_formater=wiki_page_views_data_input_formater,
    )


def count_articles():
    """Count the number of articles in a query."""

    count_n_query = """
        SELECT COUNT(*),
               SUM(CASE WHEN pvw.pageid IS NULL THEN 0 ELSE 1 END) 
        FROM article_level_info  ar
        INNER JOIN article_redirect_flag rd
            ON ar.pageid=rd.pageid
        LEFT JOIN wiki_page_view pvw
            on ar.pageid = pvw.pageid
        WHERE rd.redirect_flag = FALSE
            
        """

    return retrieve_query(count_n_query)[0]


def get_page_views_for_articles(batchsize=PAGE_VIEWS_REQUEST_BATCH):
    """Create page views for wiki database."""

    # Connect to temp db
    engine_temp_db, session_temp_db = get_connection(TEMP_DB)
    conn_temp_db = engine_temp_db.connect()

    print("Transferring pageviews already downloaded...")
    update_pageviews_already_downloaded()

    conn_temp_db.execute("DELETE FROM wiki_page_view;")

    client_user = PageviewsClient(user_agent="")

    query = """
    SELECT ar.pageid,
            ar.title
            
    FROM article_level_info  ar
    INNER JOIN article_redirect_flag rd
        ON ar.pageid=rd.pageid
    LEFT JOIN wiki_page_view pvw
        on ar.pageid = pvw.pageid
    WHERE rd.redirect_flag = FALSE
        AND pvw.pageid is NULL


    """
    # Get count of articles aready download and those to download
    n_total_to_download, n_already_downloaded = count_articles()

    first_time = True
    with tqdm(
        total=n_total_to_download,
    ) as pbar:

        for batch_articles in retrieve_query_in_batches(query, batchsize=batchsize):
            # Update articles with pageviews
            if first_time:
                pbar.update(n_already_downloaded)
                first_time = False

            # Flatten query
            _, titles_flat = zip(*batch_articles)

            # Get response with page views
            page_view_response_batch = make_pageview_request(
                client_user,
                titles_flat,
            )

            # Formate the response and insert to db
            formated_page_view_response_batch = [
                format_response_into_entry(page_view_response_batch, page_id, title)
                for page_id, title in batch_articles
            ]

            insert_articles_to_db(
                formated_page_view_response_batch, engine_temp_db, session_temp_db
            )

            # Update progressbar
            pbar.update(batchsize)
            print(pbar)


if __name__ == "__main__":
    get_page_views_for_articles(PAGE_VIEWS_REQUEST_BATCH)

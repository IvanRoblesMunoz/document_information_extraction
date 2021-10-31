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
import itertools
from pathlib import Path
import datetime
import _thread as thread
from json import JSONDecodeError

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
PAGE_VIEWS_API_TIMEOUT = 30 * 60
PAGE_VIEWS_REQUEST_BATCH = 10
PROCESS_MULTIPLIER = 1

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


# @exit_after(PAGE_VIEWS_API_TIMEOUT)
def make_pageview_request(batch_articles):
    """Make request to download pageviews."""

    client_user = PageviewsClient(user_agent="")
    _, titles_flat = zip(*batch_articles)

    return batch_articles, client_user.article_views(
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


def run_requests_concurrently(args_list):

    if __name__ == "__main__":
        pool = multiprocessing.Pool(os.cpu_count() * PROCESS_MULTIPLIER)
        response_list = pool.map(make_pageview_request, tuple(args_list))

    return response_list


# @exit_after(PAGE_VIEWS_API_TIMEOUT)
def get_page_views_for_articles(batchsize=PAGE_VIEWS_REQUEST_BATCH):
    """Create page views for wiki database."""

    # Connect to temp db
    engine_temp_db, session_temp_db = get_connection(TEMP_DB)
    conn_temp_db = engine_temp_db.connect()

    try:
        print("Transferring pageviews already downloaded...")
        update_pageviews_already_downloaded()
        print("pageviews transfered...")
    except:
        conn_temp_db.execute("DELETE FROM wiki_page_view;")

    conn_temp_db.execute("DELETE FROM wiki_page_view;")

    # client_user = PageviewsClient(user_agent="")

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

    buffer_count = 0
    args_list = []
    with tqdm(total=n_total_to_download) as pbar:
        pbar.update(n_already_downloaded)

        for batch_articles in retrieve_query_in_batches(query, batchsize=batchsize):
            # Add counter and append batch
            buffer_count += 1
            args_list.append(batch_articles)

            # try:
            # Check for buffer
            if buffer_count % (os.cpu_count() * PROCESS_MULTIPLIER) == 0:
                # Update progress bar
                pbar.update(batchsize * os.cpu_count())

                print(pbar)

                # Process requests
                response_list = run_requests_concurrently(args_list)
                if not response_list == []:
                    # Format response batches
                    formated_page_view_response_batch = []
                    for args, resp in response_list:
                        formated_page_view_response_batch += [
                            format_response_into_entry(resp, page_id, title)
                            for page_id, title in args
                        ]
                    # Insert to db
                    insert_articles_to_db(
                        formated_page_view_response_batch,
                        engine_temp_db,
                        session_temp_db,
                    )
                    # restart list
                    args_list = []
            # except:  # JSONDecodeError:

            #     args_list = []


if __name__ == "__main__":
    get_page_views_for_articles(PAGE_VIEWS_REQUEST_BATCH)


7308528 * 0.405

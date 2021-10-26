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
from pathlib import Path
import datetime
from requests import ConnectionError

from tqdm import tqdm

import numpy as np
from mwviews.api import PageviewsClient

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from src.data.wikipedia.wiki_data_base import (
    retrieve_query_in_batches,
    count_articles,
    WikiPageView,
    get_connection,
    transfer_to_new_db,
    wiki_page_views_data_input_formater,
)

# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import TEMP_DB, SQL_WIKI_DUMP

PAGE_VIEWS_START_DATE = "20201001"
PAGE_VIEWS_END_DATE = "20211001"


# =============================================================================
# Functions
# =============================================================================


def insert_articles_to_db(observations_to_insert, engine, session):
    """Insert article redirect flag into database."""
    engine.execute(WikiPageView.__table__.insert(), observations_to_insert)
    session.commit()
    session.close()


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


def recursive_call_request(client_user, titles_flat):
    try:
        page_view_response_batch = make_pageview_request(
            client_user,
            titles_flat,
        )
        return page_view_response_batch

    except ConnectionError:
        print("Retrying call again")
        client_user = PageviewsClient(user_agent="")
        return recursive_call_request(
            client_user,
            titles_flat,
        )


def update_pageviews_already_downloaded():
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


def get_page_views_for_articles(batchsize=20):

    print("Transferring pageviews already downloaded...")
    update_pageviews_already_downloaded()

    # Connect to temp db
    engine_temp_db, session_temp_db = get_connection(TEMP_DB)
    conn_temp_db = engine_temp_db.connect()
    conn_temp_db.execute("DELETE FROM wiki_page_view;")

    # TODO: check if some of these arent redirects, by removing the length check
    # 12_699_812 articles ~5.14 days
    client_user = PageviewsClient(user_agent="")

    # TODO: Check articles already added
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

    n_articles_to_download = count_articles(query)
    for batch_articles in tqdm(
        retrieve_query_in_batches(query, batchsize=batchsize),
        desc=f"Wikipedia articles batches, batchsize={batchsize}",
        total=n_articles_to_download // batchsize,
    ):

        page_ids_flat, titles_flat = zip(*batch_articles)

        # page_view_response_batch = make_pageview_request(
        #     client_user,
        #     titles_flat,
        # )
        # Make this recursively, so in case it breaks, it restarts on its own
        page_view_response_batch = recursive_call_request(client_user, titles_flat)

        formated_page_view_response_batch = [
            format_response_into_entry(page_view_response_batch, page_id, title)
            for page_id, title in batch_articles
        ]
        # for page_id, title in batch_articles:
        #     pass

        insert_articles_to_db(
            formated_page_view_response_batch, engine_temp_db, session_temp_db
        )


if __name__ == "__main__":
    get_page_views_for_articles()

# # =============================================================================
# # Imports
# # =============================================================================
# import pageviewapi

# import time

# start = time.time()
# for i in range(100):
#     pageviewapi.per_article(
#         "en.wikipedia",
#         "Barack Obama",
#         "20201001",
#         "20201001",
#         access="all-access",
#         agent="all-agents",
#         granularity="daily",
#     )
# end = time.time()
# end - start

# # Check this out
# # https://pageviews.toolforge.org/?project=en.wikipedia.org&platform=all-access&agent=user&redirects=0&start=2020-10&end=2021-09&pages=Anarchism|Barack_Obama
# import os

# os.listdir("/home/roblesi/Downloads")

# import pandas as pd

# a = pd.read_csv("/home/roblesi/Downloads/qrank.csv")
# a.sample(1)


# a = requests.get(
#     url="https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Anarchism/daily/20201010/20211012"
# )
# a = str(a.text)

# requests.request(
#     "POST",
#     url="https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Albert_Einstein/daily/2015100100/2015103100",
# )

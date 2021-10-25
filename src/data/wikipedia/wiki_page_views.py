#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:29:13 2021

@author: roblesi
"""


# =============================================================================
# Imports
# =============================================================================
import time
from mwviews.api import PageviewsClient

from src.data.wikipedia.wiki_data_base import retrieve_query

# =============================================================================
# Statics
# =============================================================================

PAGE_VIEWS_START_DATE = "20201001"
PAGE_VIEWS_END_DATE = "20201001"


# =============================================================================
# Functions
# =============================================================================


def get_page_vies_for_articles():
    p = PageviewsClient(user_agent="")


# 12_701_534 articles ~5.14 days


query = """
SELECT ar.pageid,
        ar.title
FROM article_level_info  ar
INNER JOIN article_redirect_flag rd
    ON ar.pageid=rd.pageid
WHERE rd.redirect_flag = FALSE

LIMIT 100
"""

articles = retrieve_query(query)

flat_page_ids = [item for sublist in articles for item in sublist]
page_ids_flat, titles_flat = zip(*articles)

articles_200 = p.article_views(
    "en.wikipedia",
    flat_list,
    start="20201001",
    end="20211001",
    granularity="monthly",
)
help(p)


end200 = time.time()
taken200 = end200 - start200
print(f"time taken 200 {taken200}")


query = """
SELECT title from article_level_info LIMIT 10000
"""
articles = retrieve_query(query)
flat_list = [item for sublist in articles for item in sublist]


# time taken 200 7.269313812255859
# time taken 10000 455.92031836509705


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:29:13 2021

@author: roblesi
"""

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


# =============================================================================
# Use this
# =============================================================================
from src.data.wikipedia.wiki_data_base import retrieve_query
import time
from mwviews.api import PageviewsClient

p = PageviewsClient(user_agent="<ivan.robles.munoz@hotmail.com>")

# # Fideuà vs. Paella on Spanish Wikipedia
# p.article_views(
#     "en.wikipedia",
#     ["Barack Obama", "Anarchism"],
#     start="20201001",
#     end="20211001",
#     granularity="monthly",
# )


# # Top articles on German Wikivoyage
# p.top_articles("en.wikipedia", limit=100, year=2020)

query = """
SELECT title from article_level_info LIMIT 200
"""
articles = retrieve_query(query)
flat_list = [item for sublist in articles for item in sublist]

start200 = time.time()
articles_200 = p.article_views(
    "en.wikipedia",
    flat_list,
    start="20201001",
    end="20211001",
    granularity="monthly",
)
end200 = time.time()
taken200 = end200 - start200
print(f"time taken 200 {taken200}")


query = """
SELECT title from article_level_info LIMIT 10000
"""
articles = retrieve_query(query)
flat_list = [item for sublist in articles for item in sublist]

start10000 = time.time()
articles_10000 = p.article_views(
    "en.wikipedia",
    flat_list,
    start="20201001",
    end="20211001",
    granularity="monthly",
)
end10000 = time.time()
taken10000 = end10000 - start10000
print(f"time taken 10000 {taken10000}")


query = """
SELECT title from article_level_info LIMIT 100000
"""
articles = retrieve_query(query)
flat_list = [item for sublist in articles for item in sublist]

start100000 = time.time()
articles_100000 = p.article_views(
    "en.wikipedia",
    flat_list,
    start="20201001",
    end="20211001",
    granularity="monthly",
)
end100000 = time.time()
taken100000 = end100000 - start100000
print(f"time taken 100000 {taken100000}")


10000 / 200 * 7 / 60

6e6 / 200 * 7 / 60 / 60 / 24
6e6 / 10000 * 455 / 60 / 60 / 24

(456 / 10000) / (7.26 / 200)

# time taken 200 7.269313812255859
# time taken 10000 455.92031836509705

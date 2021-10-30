#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 19:10:02 2021

@author: roblesi
"""

# =============================================================================
# Imports
# =============================================================================
import os
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle
import multiprocessing
import re


from tqdm import tqdm

from haystack import Document
from haystack.preprocessor import PreProcessor
from src.data.wikipedia.wiki_data_base import retrieve_query_in_batches, retrieve_query
from src.retriever.database_temp_faiss import insert_existing_faiss_articles

# =============================================================================
# Statics
# =============================================================================
# from src.data.data_statics import SQL_WIKI_DUMP # TODO: Change this
from src.retriever.retriever_statics import (
    FAISS_TEMP_SQL_DB_PATH,
    FAISS_GEN_N_ARTICLES_BATCH,
)

# TODO: Remove this
SQL_WIKI_DUMP = "/home/roblesi/git/document_information_extraction/data/interim/wiki_db_dumps_for_faiss.db"

REDIRECT_RE = re.compile("REDIRECT")


# =============================================================================
# Utility functions
# =============================================================================


def row_to_article(article):
    """Decodes row from SQL query to produce summary and body."""
    page_id = article[0]
    section_titles = pickle.loads(article[1])
    summary = article[2]
    body_sections = pickle.loads(article[3])
    title = article[4]
    return page_id, section_titles, summary, body_sections, title


def is_redirect(summary, body_sections):
    """Check if article is a redirect."""
    body_is_list = body_sections == []
    contains_redirect = bool(re.search(REDIRECT_RE, summary))
    small_summary = len(summary.split()) <= 20
    return body_is_list & contains_redirect & small_summary


def count_articles(query, out_f=SQL_WIKI_DUMP):
    """Count the number of articles in a query."""

    count_n_query = (
        """
        SELECT COUNT(*) AS all_articles,
               SUM(CASE WHEN faipageid IS NULL THEN 0 ELSE 1 END) AS articles_in
        
        FROM
        """
        + " "
        + "FROM".join(query.split("FROM")[1:])
    )

    count_n_query = count_n_query.split("WHERE sub.faipageid IS NULL")[0]

    return retrieve_query(count_n_query, out_f=out_f)[0]


def article_sub_query(n_sample_articles):
    """Make article subquery."""
    sub_query = """
    SELECT  wk.pageid
           ,wk.section_titles
           ,wk.summary
           ,wk.body_sections
           ,ar.title
           ,fai.pageid AS faipageid
    
           
    FROM wiki_articles wk
    
    INNER JOIN article_level_info ar
        ON ar.pageid = wk.pageid
    
    INNER JOIN article_redirect_flag rd
        ON ar.pageid=rd.pageid
    
    LEFT JOIN articles_in_faiss fai
        on wk.pageid = fai.pageid
    
    LEFT JOIN wiki_page_view pvw
        on ar.pageid = pvw.pageid
    
    WHERE rd.redirect_flag = FALSE
        AND NOT pvw.mean_views IS NULL
    
    ORDER BY pvw.mean_views DESC
    """
    if n_sample_articles:
        sub_query += f"LIMIT {n_sample_articles}"

    return sub_query


def article_filter_query(n_sample_articles):
    """Return query to prioritise article embedding."""
    query = (
        """
    SELECT *
    FROM (
    """
        + article_sub_query(n_sample_articles)
        + """
        ) sub
    WHERE sub.faipageid IS NULL 
    
    """
    )
    return query


def sql_reader_passage_generator(
    storage_method: str,
    n_sample_articles: int = None,
    out_f: str = SQL_WIKI_DUMP,
    batchsize: int = 1,
    skip_redirects: bool = True,
) -> dict:
    """
    Generate passages from articles.

    Splits the articles into passages using the summary body sections as
    passages.

    Parameters
    ----------
    storage_method : str
        Storage method used, options are "faiss" or "elasticsearch".
        elasticsearch will pass the whole article, faiss will split the
        article into passages based on article sections.
    n_sample_articles : int, optional
        Number of sample articles to limit query, if None, all are returned.
        The default is None.
    out_f : str, optional
        Location of sql database. The default is SQL_WIKI_DUMP.
    batchsize : int, optional
        The default is 5.
    skip_redirects : bool, optional
        If to skip redirects. The default is True.

    Yields
    ------
    dict
        1. For elasticsearch
           {"page_id": page_id,
            "title": title,
            "section_title": section_title,
            "section_text": section_text,}
        2. For Faiss
           {"page_id": page_id,
            "title": title,
            "section_text": article}

    """
    # Update table to check if we have already inserted certain aricles
    if os.path.isfile(FAISS_TEMP_SQL_DB_PATH):
        insert_existing_faiss_articles()

    # Define query
    query = article_filter_query(n_sample_articles)

    print("Counting articles...")
    n_all_articles, articles_in = count_articles(query)

    print("Running query for generator...")
    article_generator = retrieve_query_in_batches(
        query, out_f=out_f, batchsize=batchsize
    )

    # Iterate through query
    for article_batch in tqdm(
        article_generator,
        desc=f"Wikipedia articles, batchsize={batchsize}, already in={articles_in}, {articles_in/n_all_articles*100:.2f}%",
        total=(n_all_articles - articles_in) // batchsize,
    ):

        # Iterate through batch
        for article in article_batch:
            page_id, section_titles, summary, body_sections, title = row_to_article(
                article
            )
            # If we want to skip redirects ignore else yield passages
            if skip_redirects & is_redirect(summary, body_sections):
                pass
            else:
                # If Faiss, we will yield a passage for each section
                if storage_method == "faiss":
                    # Define summary document
                    summary = [
                        {
                            "page_id": page_id,
                            "title": title,
                            "section_title": "summary",
                            "section_text": summary,
                        }
                    ]

                    # Define body document
                    body = [
                        {
                            "page_id": page_id,
                            "title": title,
                            "section_title": sect_title_,
                            "section_text": sect_body_,
                        }
                        for sect_title_, sect_body_ in zip(
                            section_titles, body_sections
                        )
                    ]
                    # Yield both together
                    yield summary + body

                # If elastic search, we will yield a single
                elif storage_method == "elasticsearch":
                    yield {
                        "page_id": page_id,
                        "title": title,
                        "section_text": summary + "".join(body_sections),
                    }
                else:
                    raise Exception(
                        f""""{storage_method}" is not a supported storage \
                        method, please use "faiss" or "elasticsearch"."""
                    )


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def make_document_from_passage(passage: dict, storage_method: str) -> dict:
    """Convert passage to compatible storage format."""
    document = {
        "content": passage["section_text"],
        "meta": {
            "page_id": passage["page_id"],
            "title": passage["title"],
        },
    }

    # If storage method is faiss add section title
    if storage_method == "faiss":
        document["meta"]["section_title"] = passage["section_title"]

    elif storage_method == "elasticsearch":
        pass
    else:
        raise Exception(
            f""""{storage_method}" is not a supported storage \
             method, please use "faiss" or "elasticsearch"."""
        )
    return document


def process_article(args):
    """Process article function for multiprocessing."""
    article, preprocessor, storage_method = args
    split_docs = preprocessor.process(article)
    return split_docs


def processed_document_generator(
    storage_method: str,
    preprocessor: PreProcessor,
    batch_size_doc_generator: int = FAISS_GEN_N_ARTICLES_BATCH,
    **kwargs,
) -> list:

    """
    Split documents in a suitable format and batchsize.

    Parameters
    ----------
    storage_method : str
        Storage method used, options are "faiss" or "elasticsearch".
        elasticsearch will pass the whole article, faiss will split the
        article into passages based on article sections.
    preprocessor : PreProcessor
        Preprocessor used to process the documents.
    batch_size_doc_generator : int, optional
        DESCRIPTION. The default is 10000.
    **kwargs : dict, optional
        Kwargs, this are also passed to sql_reader_passage_generator.

    Yields
    ------
    list
        DESCRIPTION.

    """

    # Update kwargs for sql reader generator
    kwargs["storage_method"] = storage_method

    counter = 0
    article_batch = []

    for article in sql_reader_passage_generator(**kwargs):

        article_processed = [
            make_document_from_passage(passage, storage_method) for passage in article
        ]
        article_batch.append((article_processed, preprocessor, storage_method))

        counter += 1
        # If batch finished, yield batch
        if counter % batch_size_doc_generator == 0:

            # Suppress output from preprocessor
            with suppress_stdout_stderr():
                pool = multiprocessing.Pool(os.cpu_count())
                split_docs = pool.map(process_article, article_batch)

                # split_docs = [
                #     preprocessor.process(article) for article in article_batch
                # ]
            article_batch = []

            # Modify document type for faiss compatibility
            if storage_method == "faiss":
                split_docs = [
                    [Document(**i) for i in split_articles]
                    for split_articles in split_docs
                ]

            yield split_docs

    # yield remaing passages asthe last batch
    if len(article_batch) > 0:
        # Suppress output from preprocessor
        with suppress_stdout_stderr():
            pool = multiprocessing.Pool(os.cpu_count())
            split_docs = pool.map(process_article, article_batch)

        # Modify document type for faiss compatibility
        if storage_method == "faiss":
            split_docs = [
                [Document(**i) for i in split_articles] for split_articles in split_docs
            ]

        yield split_docs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 19:10:02 2021

@author: roblesi
"""
# =============================================================================
# Imports
# =============================================================================
import pickle
import re
import sqlite3

from tqdm import tqdm

from haystack import Document
from haystack.preprocessor import PreProcessor
from src.data.wikipedia.wiki_data_base import retrieve_query_in_batches, retrieve_query

# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import SQL_WIKI_DUMP
from src.retriever.retriever_statics import FAIIS_DB_DATA_PATH

# from src.retriever.retriever_statics import FAISS_MAX_PASSAGE_TOKEN_LEN

REDIRECT_RE = re.compile("REDIRECT")
# =============================================================================
# Check data already in
# =============================================================================
def connect_to_faiss_db():
    """Connects to faiss db."""
    conn = sqlite3.connect(FAIIS_DB_DATA_PATH.replace("sqlite:///", ""))
    return conn


def retrieve_tables_in_database(conn):
    """Retrieve table names."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return tables


def retrieve_article_page_ids_already_in_faiss(conn):
    """Retrieve page ids that have already being inserted."""
    cursor = conn.cursor()

    query = """
            WITH cte(c1, c2, c3, meta_data_label, meta_data_value, c6) AS (
                SELECT * FROM meta_document
            
            )
            SELECT DISTINCT meta_data_value
            FROM cte
            WHERE meta_data_label = "page_id";
            """
    cursor.execute(query)
    existing_ids = cursor.fetchall()
    return existing_ids


def insert_existing_faiss_articles():
    conn = connect_to_faiss_db()
    existing_ids = retrieve_article_page_ids_already_in_faiss(conn)


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


def count_articles(query):
    """Count the number of articles in a query."""

    count_n_query = (
        """
        SELECT COUNT(*)
        FROM
        """
        + " "
        + query.split("FROM")[1]
    )

    return retrieve_query(count_n_query)[0][0]


def sql_reader_passage_generator(
    storage_method: str,
    n_sample_articles: int = None,
    out_f: str = SQL_WIKI_DUMP,
    batchsize: int = 10,
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

    # Define query
    query = """
    SELECT  wk.pageid
           ,wk.section_titles
           ,wk.summary
           ,wk.body_sections
           ,ar.title
    FROM wiki_articles wk
    INNER JOIN article_level_info ar
        ON ar.pageid = wk.pageid
    """

    if n_sample_articles:
        query += f"LIMIT {n_sample_articles}"
    else:
        n_sample_articles = count_articles(query)

    # Iterate through query
    for article_batch in tqdm(
        retrieve_query_in_batches(query, out_f=out_f, batchsize=batchsize),
        desc=f"Wikipedia articles batches, batchsize={batchsize}",
        total=n_sample_articles // batchsize,
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
                    yield {
                        "page_id": page_id,
                        "title": title,
                        "section_title": "summary",
                        "section_text": summary,
                    }
                    for sect_title_, sect_body_ in zip(section_titles, body_sections):
                        yield {
                            "page_id": page_id,
                            "title": title,
                            "section_title": sect_title_,
                            "section_text": sect_body_,
                        }
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


def processed_document_generator(
    storage_method: str,
    preprocessor: PreProcessor,
    batch_size_doc_generator: int = 1000,
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
    passage_batch = []

    for passage in sql_reader_passage_generator(**kwargs):
        passage_batch.append(make_document_from_passage(passage, storage_method))

        counter += 1
        # If batch finished, yield batch
        if counter % batch_size_doc_generator == 0:
            split_docs = preprocessor.process(passage_batch)
            passage_batch = []

            # Modify document type for faiss compatibility
            if storage_method == "faiss":
                split_docs = [Document(**i) for i in split_docs]

            yield split_docs

    # yield remaing passages asthe last batch
    if len(passage_batch) > 0:
        split_docs = preprocessor.process(
            passage_batch,
        )
        passage_batch = []

        # Modify document type for faiss compatibility
        if storage_method == "faiss":
            split_docs = split_docs = [Document(**i) for i in split_docs]

        yield split_docs

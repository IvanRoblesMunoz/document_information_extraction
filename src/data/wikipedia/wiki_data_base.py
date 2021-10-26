#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 01:26:02 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import os
import sqlite3
import itertools

from sqlalchemy import (
    create_engine,
    Boolean,
    Column,
    Integer,
    Text,
    LargeBinary,
    # ForeignKey,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import SQL_WIKI_DUMP
from src.data.data_statics import (
    MIN_TOKENS_SUMMARY,
    MAX_TOKENS_SUMMARY,
    MIN_TOKENS_BODY,
    MIN_COMPRESION_RATIO,
    MAX_COMPRESION_RATIO,
)

# =============================================================================
# Input
# =============================================================================
Base = declarative_base()


class WikiArticles(Base):
    """Article database."""

    __tablename__ = "wiki_articles"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    section_title = Column("section_titles", LargeBinary, unique=False)
    summary = Column("summary", Text, unique=False)
    body_sections = Column("body_sections", LargeBinary, unique=False)
    section_word_count = Column("section_word_count", LargeBinary, unique=False)


class ArticleLevelInfo(Base):
    """Article database."""

    __tablename__ = "article_level_info"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    title = Column("title", Text, unique=False)
    summary_word_count = Column("summary_word_count", Integer, unique=False)
    body_word_count = Column("body_word_count", Integer, unique=False)


class ArticleRedirectFlag(Base):
    """Database indicating which articles are redirects."""

    __tablename__ = "article_redirect_flag"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    redirect_flag = Column("redirect_flag", Boolean, unique=False)


class WikiPageView(Base):
    """Article database."""

    __tablename__ = "wiki_page_view"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    title = Column("title", Text, unique=False)
    pageviews_2020_10_01 = Column("pageviews_2020_10_01", Integer, unique=False)
    pageviews_2020_11_01 = Column("pageviews_2020_11_01", Integer, unique=False)
    pageviews_2020_12_01 = Column("pageviews_2020_12_01", Integer, unique=False)
    pageviews_2021_01_01 = Column("pageviews_2021_01_01", Integer, unique=False)
    pageviews_2021_02_01 = Column("pageviews_2021_02_01", Integer, unique=False)
    pageviews_2021_03_01 = Column("pageviews_2021_03_01", Integer, unique=False)
    pageviews_2021_04_01 = Column("pageviews_2021_04_01", Integer, unique=False)
    pageviews_2021_05_01 = Column("pageviews_2021_05_01", Integer, unique=False)
    pageviews_2021_06_01 = Column("pageviews_2021_06_01", Integer, unique=False)
    pageviews_2021_07_01 = Column("pageviews_2021_07_01", Integer, unique=False)
    pageviews_2021_08_01 = Column("pageviews_2021_08_01", Integer, unique=False)
    pageviews_2021_09_01 = Column("pageviews_2021_09_01", Integer, unique=False)
    mean_views = Column("mean_views", Float, unique=False)


class WikiArticleNovelty(Base):
    """Article novelty database."""

    __tablename__ = "wiki_article_novelty"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    novelty_tokens = Column("novelty_tokens", Text, unique=False)
    novelty_bigrams = Column("novelty_bigrams", Integer, unique=False)
    novelty_trigrams = Column("novelty_trigrams", Integer, unique=False)


class WikiCosineSimilarity(Base):
    """Article cosine similarty."""

    __tablename__ = "wiki_article_cosine_similarity"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Text, primary_key=True)
    semantic_similarity = Column("semantic_similarity", Float, unique=False)


class ArticlesInFAISS(Base):
    """List of articles already in FAISS db during embedding process."""

    __tablename__ = "articles_in_faiss"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)


class FAISSEmbeddingStore(Base):
    """FAISS embedding store"""

    __tablename__ = "faiss_embedding_store"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    title = Column("title", Text, unique=False)
    embeddings = Column("embeddings", LargeBinary)
    body_sections = Column("body_sections", LargeBinary, unique=False)


def get_connection(out_f=SQL_WIKI_DUMP):
    """Get connection to database."""
    engine = create_engine(f"sqlite:///{str(out_f)}", echo=True)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)
    session = session()
    return engine, session


def create_wiki_data_base(queue_sql, out_f=SQL_WIKI_DUMP):
    """Create wiki SQL database from iterable."""

    # If database exists delete and create again
    if os.path.exists(out_f):
        os.remove(out_f)

    engine, session = get_connection(out_f=out_f)

    # Get arguments
    sql_args = queue_sql.get()

    # Insert stuff
    while sql_args is not None:

        # separate into text and info
        section_level_output_list, article_level_output_list = sql_args

        print("-" * 50)
        # Insert text data
        engine.execute(WikiArticles.__table__.insert(), section_level_output_list)
        session.commit()
        session.close()

        # Insert text data
        engine.execute(ArticleLevelInfo.__table__.insert(), article_level_output_list)
        session.commit()
        session.close()
        # insert artcle data

        # Get next batch
        sql_args = queue_sql.get()


def insert_observations_in_table_mp(queue_sql, table, out_f=SQL_WIKI_DUMP):
    """
    Insert observations into table using  multiprocessing.

    It will delete the database if it exists.
    """
    # If database exists delete and create again
    if os.path.exists(out_f):
        os.remove(out_f)

    engine, session = get_connection(out_f=out_f)
    # Get arguments
    sql_args = queue_sql.get()

    while sql_args is not None:
        print("-" * 50)
        # Insert text data
        engine.execute(table.__table__.insert(), sql_args)
        session.commit()
        session.close()

        # Get next batch
        sql_args = queue_sql.get()


# =============================================================================
# Output
# =============================================================================


def retrieve_query(query: tuple, out_f: str = SQL_WIKI_DUMP):
    """Retrieve query from database."""
    conn = sqlite3.connect(str(out_f))
    cur = conn.cursor()
    if type(query) == str:
        cur.execute(query)
    else:
        cur.execute(*query)
    rows = cur.fetchall()
    return rows


def retrieve_query_in_batches(
    query: tuple, out_f: str = SQL_WIKI_DUMP, batchsize: int = 1000
):
    """Retrieve query from database in batches."""
    conn = sqlite3.connect(out_f)
    cur = conn.cursor()
    if type(query) == str:
        cur.execute(query)
    else:
        cur.execute(*query)
    while True:
        batch = cur.fetchmany(batchsize)
        if not batch:
            break

        yield batch


def retrive_suitable_strings(
    out_f: str = SQL_WIKI_DUMP,
    limit: int = None,
    min_tokens_summary: int = MIN_TOKENS_SUMMARY,
    max_tokens_summary: int = MAX_TOKENS_SUMMARY,
    max_tokens_body: int = int(MAX_TOKENS_SUMMARY / MIN_COMPRESION_RATIO),
    min_tokens_body: int = MIN_TOKENS_BODY,
    min_ratio: float = MIN_COMPRESION_RATIO,
    max_ratio: float = MAX_COMPRESION_RATIO,
    batchsize: int = 1,
) -> list:
    """Obtain a list of article ids based on character length of summary and body."""
    query = f"""
            SELECT wk.*
            FROM article_level_info ar
            INNER JOIN wiki_articles wk 
                ON ar.pageid = wk.pageid
            WHERE body_word_count>={min_tokens_body} 
                AND body_word_count<={max_tokens_body}
                AND summary_word_count>={min_tokens_summary}
                AND summary_word_count<={max_tokens_summary}
                AND CAST( summary_word_count AS FLOAT)/ CAST( body_word_count AS FLOAT) >= {min_ratio}
                AND CAST( summary_word_count AS FLOAT)/CAST( body_word_count AS FLOAT) <= {max_ratio}
               
            """
    if not limit is None:
        query += f"LIMIT {limit}"

    for rows in retrieve_query_in_batches(query, out_f, batchsize=batchsize):

        yield rows


def retrive_observations_from_ids(
    ids,
    out_f=SQL_WIKI_DUMP,
    table="wiki_articles",
    id_column="pageid",
    chunksize=10000,
):
    """Retrieve pageid, body and summary based on list of ids."""

    def _retrive_single_query(batch_ids, out_f):
        """Retrieve single query batch."""

        query = (
            f"""
            SELECT *
            FROM {table}
            WHERE {id_column} in ({','.join(['?']*len(batch_ids))})
            """,
            batch_ids,
        )
        return retrieve_query(query, out_f)

    iterations = len(ids) // chunksize + 1
    relevant_obs = []
    for i in range(iterations):
        obs = _retrive_single_query(ids[chunksize * i : chunksize * (i + 1)], out_f)
        relevant_obs.append(obs)

    relevant_obs = list(itertools.chain(*relevant_obs))
    return relevant_obs


# =============================================================================
# Transfer
# =============================================================================
def redirect_flag_data_input_formater(batch):
    """Formats data produced by generator for redirect flag to insert in db."""
    return [
        {
            "pageid": obs[0],
            "redirect_flag": obs[1],
        }
        for obs in batch
    ]


def wiki_page_views_data_input_formater(batch):
    """Formats data produced by generator for wiki page_views to insert in db."""
    return [
        {
            "pageid": obs[0],
            "title": obs[1],
            "pageviews_2020_10_01": obs[2],
            "pageviews_2020_11_01": obs[3],
            "pageviews_2020_12_01": obs[4],
            "pageviews_2021_01_01": obs[5],
            "pageviews_2021_02_01": obs[6],
            "pageviews_2021_03_01": obs[7],
            "pageviews_2021_04_01": obs[8],
            "pageviews_2021_05_01": obs[9],
            "pageviews_2021_06_01": obs[10],
            "pageviews_2021_07_01": obs[11],
            "pageviews_2021_08_01": obs[12],
            "pageviews_2021_09_01": obs[13],
            "mean_views": obs[14],
        }
        for obs in batch
    ]


def novelty_data_input_formater(batch):
    """Formats data produced by generator for novelty data to insert in db."""
    return [
        {
            "pageid": obs[0],
            "novelty_tokens": obs[1],
            "novelty_bigrams": obs[2],
            "novelty_trigrams": obs[3],
        }
        for obs in batch
    ]


def semantic_similarity_data_input_formater(batch):
    """Formats data produced by generator for novelty data to insert in db."""
    return [
        {
            "pageid": obs[0],
            "semantic_similarity": obs[1],
        }
        for obs in batch
    ]


def insert_into_db(generator, table, dest_db, batch_formater):
    """Insert data into database without deleting original table or db."""
    # Connect
    engine, session = get_connection(dest_db)

    # Insert all
    for batch in generator:
        # Insert text data
        engine.execute(table.__table__.insert(), batch_formater(batch))
        session.commit()
        session.close()


def transfer_to_new_db(
    src_query, src_db, dest_db, dest_table, batch_formater, batchsize=10000
):
    """Transfers data from one database to another."""
    # Get data we want to transfer
    transfer_generator = retrieve_query_in_batches(
        query=src_query,
        out_f=src_db,
        batchsize=batchsize,
    )

    # Insert data we want to insert
    insert_into_db(
        transfer_generator,
        dest_table,
        dest_db,
        batch_formater=batch_formater,
    )


# =============================================================================
# Other
# =============================================================================
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

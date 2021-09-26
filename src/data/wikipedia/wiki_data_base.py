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

from sqlalchemy import (
    create_engine,
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
    title = Column("title", Float, unique=False)
    summary_word_count = Column("summary_word_count", Float, unique=False)
    body_word_count = Column("body_word_count", Float, unique=False)


class WikiArticleNovelty(Base):
    """Article database."""

    __tablename__ = "wiki_article_novelty"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    novelty_tokens = Column("novelty_tokens", Text, unique=False)
    novelty_bigrams = Column("novelty_bigrams", Integer, unique=False)
    novelty_trigrams = Column("novelty_trigrams", Integer, unique=False)


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
    conn = sqlite3.connect(out_f)
    cur = conn.cursor()
    if type(query) == str:
        cur.execute(query)
    else:
        cur.execute(*query)
    rows = cur.fetchall()
    return rows


def retrieve_query_in_batches(
    query: tuple, out_f: str = SQL_WIKI_DUMP, batchsize: int = 1
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
    min_tokens_body: int = MIN_TOKENS_BODY,
    min_ratio: float = MIN_COMPRESION_RATIO,
    max_ratio: float = MAX_COMPRESION_RATIO,
    retrieve_method: str = "Full",
    batchsize: int = 1,
) -> list:
    """Obtain a list of article ids based on character length of summary and body."""

    query = f"""
            SELECT wk.*
            FROM article_level_info ar
            LEFT JOIN wiki_articles wk 
                ON ar.pageid = wk.pageid
            WHERE body_word_count>={min_tokens_body} 
                AND summary_word_count>={min_tokens_summary}
                AND summary_word_count<={max_tokens_summary}
                AND CAST( summary_word_count AS FLOAT)/ CAST( body_word_count AS FLOAT) >= {min_ratio}
                AND CAST( summary_word_count AS FLOAT)/CAST( body_word_count AS FLOAT) <= {max_ratio}
            """
    if not limit is None:
        query += f"LIMIT {limit}"

    if retrieve_method == "Full":
        rows = retrieve_query(query, out_f)
        return rows

    elif retrieve_method == "Batches":
        for rows in retrieve_query_in_batches(query, out_f, batchsize=batchsize):
            yield rows
    else:
        raise Exception(
            f"{retrieve_method} not supported, please use 'Full' or 'Batches'"
        )


# =============================================================================
# Transfer
# =============================================================================


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


# import sys

# def open_db(db):
#     conn = sqlite3.connect(db)
#     # Let rows returned be of dict/tuple type
#     conn.row_factory = sqlite3.Row

#     return conn

# def copy_table(table, src, dest,batchsize=10000):
#     print(f"Copying table: {table} from {src} to {dest}")
#     sc = src.execute(f'SELECT * FROM %s' % table)
#     ins = None
#     dc = dest.cursor()
#     for row in sc.fetchmany(batchsize):
#         if not ins:
#             cols = tuple([k for k in row.keys() if k != 'id'])
#             ins = 'INSERT OR REPLACE INTO %s %s VALUES (%s)' % (table, cols,
#                                                      ','.join(['?'] * len(cols)))
#             print 'INSERT stmt = ' + ins
#         c = [row[c] for c in cols]
#         dc.execute(ins, c)

#     dest.commit()

# src_conn  = open_db(sys.argv[1])
# dest_conn = open_db(sys.argv[2])

# copy_table('audit', src_conn, dest_conn)


# def retrive_observations_from_ids(
#     ids,
#     out_f=SQL_WIKI_DUMP,
#     # table="wiki_articles",
#     id_column="page_id",
#     chunksize=10000,
# ):
#     """Retrieve pageid, body and summary based on list of ids."""

#     def _retrive_single_query(batch_ids, out_f):
#         """Retrieve single query batch."""

#         query = (
#             f"""
#             SELECT pageid,summary,body
#             FROM wiki_articles
#             WHERE {id_column} in ({','.join(['?']*len(batch_ids))})
#             """,
#             batch_ids,
#         )
#         return retrieve_query(query, out_f)

#     iterations = len(ids) // chunksize + 1
#     relevant_obs = []
#     for i in range(iterations):
#         obs = _retrive_single_query(ids[chunksize * i : chunksize * (i + 1)], out_f)
#         relevant_obs.append(obs)

#     relevant_obs = list(itertools.chain(*relevant_obs))
#     return relevant_obs


# query = """
# SELECT wk.*
# FROM article_level_info ar
# LEFT JOIN wiki_articles wk
#     ON ar.pageid = wk.pageid
# WHERE ar.body_word_count>15 and ar.summary_word_count>150
# LIMIT 25

# """
# import pickle

# data = retrieve_query(query)
# for row in data:
#     pageid = row[0]
#     section_titles = pickle.loads(row[1])
#     summary = row[2]
#     section_word_count = pickle.loads(row[3])
#     body_sections = pickle.loads(row[4])

#     check = pageid, section_titles, summary, section_word_count, body_sections

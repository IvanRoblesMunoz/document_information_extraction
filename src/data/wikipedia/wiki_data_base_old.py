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

# import time
import itertools

import sqlite3

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Text,
    LargeBinary,
    ForeignKey,
    Float,
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker  # , relationship


from src.data.data_statics import (
    SQL_WIKI_DUMP,
    MIN_TOKENS_SUMMARY,
    MIN_SUMMARY_RATIO,
    MAX_SUMMARY_RATIO,
    MAX_TOKENS_BODY,
)

# =============================================================================
# Input
# =============================================================================
Base = declarative_base()


class User(Base):
    """Article database."""

    __tablename__ = "wiki_articles"
    __table_args__ = {"extend_existing": True}

    pageid = Column("pageid", Integer, primary_key=True)
    title = Column("title", Text, unique=False)
    summary = Column("summary", LargeBinary, unique=False)
    body = Column("body", LargeBinary, unique=False)
    n_characters_summary = Column("n_characters_summary", Integer, unique=False)
    n_characters_body = Column("n_characters_body", Integer, unique=False)


class SummaryLength(Base):
    """Summary and body length database."""

    __tablename__ = "article_length"
    __table_args__ = {"extend_existing": True}

    pageid = Column(
        "pageid", Integer, ForeignKey("wiki_articles.pageid"), primary_key=True
    )
    n_sent_summary = Column("n_sent_summary", Integer, primary_key=False)
    n_sent_body = Column("n_sent_body", Integer, primary_key=False)
    n_tokens_summary = Column("n_tokens_summary", Integer, primary_key=False)
    n_tokens_text = Column("n_tokens_text", Integer, primary_key=False)

    user = relationship(User, uselist=False)


class SummarySimilarity(Base):
    """Summary similarity metrics database."""

    __tablename__ = "summary_similarity"
    __table_args__ = {"extend_existing": True}

    sentence_id = Column("sentence_id", Text, primary_key=True)
    pageid = Column(
        "pageid", Integer, ForeignKey("wiki_articles.pageid"), primary_key=False
    )
    sentence_num = Column("sentence_num", Integer, primary_key=False)
    precision_bert_score = Column("precision_bert_score", Float, primary_key=False)
    recall_bert_score = Column("recall_bert_score", Float, primary_key=False)
    f1_bert_score = Column("f1_bert_score", Float, primary_key=False)
    sentence = Column("sentence", Text, primary_key=False)

    user = relationship(User, uselist=True)


def get_connection(out_f=SQL_WIKI_DUMP):
    """Get connection to database."""
    engine = create_engine(f"sqlite:///{str(out_f)}", echo=True)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    # Insert stuff
    session = Session()
    return engine, session


def create_wiki_data_base(queue_sql, out_f=SQL_WIKI_DUMP):
    """Create wiki SQL database from iterable."""

    # If database exists delete and create again
    if os.path.exists(out_f):
        os.remove(out_f)

    engine, session = get_connection(out_f=out_f)

    # Get arguments
    args = queue_sql.get()

    # Insert stuff
    while args is not None:
        print("-" * 50)
        engine.execute(
            User.__table__.insert(),
            [
                {
                    "pageid": arg[0],
                    "title": arg[1],
                    "summary": arg[2],
                    "body": arg[3],
                    "n_characters_summary": arg[4],
                    "n_characters_body": arg[5],
                }
                for arg in args
            ],
        )
        session.commit()
        session.close()

        # Get next batch
        args = queue_sql.get()


def create_text_length_database(queue_sql, out_f=SQL_WIKI_DUMP):
    """Create database containing summary and body length."""
    # Connect to database
    engine, session = get_connection(out_f=out_f)

    # Get data
    args = queue_sql.get()

    # Insert into table
    while args is not None:
        print("-" * 50)
        engine.execute(
            SummaryLength.__table__.insert(),
            [
                {
                    "pageid": arg[0],
                    "n_sent_summary": arg[1],
                    "n_sent_body": arg[2],
                    "n_tokens_summary": arg[3],
                    "n_tokens_text": arg[4],
                }
                for arg in args
            ],
        )

        session.commit()
        session.close()

        # Get next batch
        args = queue_sql.get()


def create_summary_sentence_similarity_database(final_results, out_f=SQL_WIKI_DUMP):
    """Create database containing sentence similarity metrics."""
    # Connect to database
    try:
        engine, session = get_connection(out_f=out_f)

        # Get data

        # print("-" * 50)
        engine.execute(
            SummarySimilarity.__table__.insert(),
            [
                {
                    "sentence_id": arg[0],
                    "pageid": arg[1],
                    "sentence_num": arg[2],
                    "precision_bert_score": arg[3],
                    "recall_bert_score": arg[4],
                    "f1_bert_score": arg[5],
                    "sentence": arg[6],
                }
                for arg in final_results
            ],
        )

        session.commit()
        session.close()
    except KeyboardInterrupt:
        session.commit()
        session.close()


# =============================================================================
# Retrieval
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


import time

query = """
SELECT Count(*)
FROM article_level_info
WHERE body_word_count>250 and summary_word_count>50
"""


while True:
    print("-" * 15)
    for b, s in [(500, 100), (250, 50), (0, 0)]:
        query = f"""
                SELECT Count(*)
                FROM article_level_info
                WHERE body_word_count>={b} and summary_word_count>={s}
                """
        rows = retrieve_query(query)
        print(b, s, rows)
    time.sleep(300)


def retrive_suitable_column_ids(
    out_f: str = SQL_WIKI_DUMP,
    min_summary: int = MIN_TOKENS_SUMMARY,
    max_body: int = MAX_TOKENS_BODY,
    min_ratio: float = MIN_SUMMARY_RATIO,
    max_ratio: float = MAX_SUMMARY_RATIO,
    limit_ids: int = None,
) -> list:
    """Obtain a list of article ids based on character length of summary and body."""
    query = f"""
        SELECT pageid
        FROM article_length 
        WHERE n_tokens_summary >= {min_summary}
            AND n_tokens_text <= {max_body}
            AND CAST( n_tokens_summary AS FLOAT)/ CAST( n_tokens_text AS FLOAT) >= {min_ratio}
            AND CAST( n_tokens_summary AS FLOAT)/CAST( n_tokens_text AS FLOAT) <= {max_ratio}
    
        """
    if not limit_ids is None:
        query += f"LIMIT {limit_ids}"
    rows = retrieve_query(query, out_f)
    suitable_entries = [page_id[0] for page_id in rows]
    return suitable_entries


def retrive_observations_from_ids(
    ids,
    out_f=SQL_WIKI_DUMP,
    # table="wiki_articles",
    id_column="page_id",
    chunksize=10000,
):
    """Retrieve pageid, body and summary based on list of ids."""

    def _retrive_single_query(batch_ids, out_f):
        """Retrieve single query batch."""

        query = (
            f"""
            SELECT pageid,summary,body
            FROM wiki_articles
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


# if __name__ == "__main__":
#     ids = retrive_suitable_column_ids()
#     print(f"n_suitable: {len(ids)}")
# relevant_obs = retrive_observations_from_ids(ids[0:1], SQL_WIKI_DUMP)

# query = f"""
#       SELECT title
#       FROM wiki_articles

#       """


# rows = [i[0] for i in retrieve_query(query)]
# rows.sort()
# 3299560

# # [i for i in rows if "archism" in i]

# 5.3e6 / 3299560 * 1120 / 60
# 3299560 / 5.3e6

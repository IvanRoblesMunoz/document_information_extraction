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

    pageid = Column(
        "pageid", Integer, ForeignKey("wiki_articles.pageid"), primary_key=True
    )
    title = Column("title", Text, unique=False)
    summary_word_count = Column("summary_word_count", Integer, unique=False)
    body_word_count = Column("body_word_count", Integer, unique=False)

    wiki_article = relationship(WikiArticles, uselist=False)


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

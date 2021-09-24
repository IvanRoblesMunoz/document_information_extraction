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


class WikiArticles(Base):
    """Article database."""

    __tablename__ = "wiki_articles"
    __table_args__ = {"extend_existing": True}

    key = Column("key", Integer, primary_key=True, autoincrement=True)
    pageid = Column("pageid", Integer, unique=False)
    section_title = Column("section_title", Text, unique=False)
    section_text = Column("section_text", Text, unique=False)
    section_word_count = Column("section_word_count", Integer, unique=False)


class ArticleLevelInfo(Base):
    """Article database."""

    __tablename__ = "article_level_info"
    __table_args__ = {"extend_existing": True}

    pageid = Column(
        "pageid", Integer, primary_key=True  # ForeignKey("wiki_articles.pageid"),
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

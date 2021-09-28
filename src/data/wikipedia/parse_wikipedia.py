#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:39:46 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import time
import pickle
from datetime import timedelta
import os
import sys
from pathlib import Path
import re
import bz2
from multiprocessing import Process, Queue
from itertools import zip_longest
from tqdm import tqdm
import nltk
from gensim.corpora.wikicorpus import iterparse, filter_wiki, remove_markup

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))


from src.data.wikipedia.wiki_data_base import create_wiki_data_base

# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import (
    RAW_WIKIPEDIA_CORPUS,
    READ_QUE_SIZE,
    SQL_QUE_SIZE,
    N_PROCESSES,
    BATCH_SIZE,
)


KEYWORDS = [
    "notes",
    "sources",
    "primary sources",
    "secondary sources",
    "further reading",
    "external links",
    "see also",
    "References",
]
REMOVE_REF = "(?i)" + "|".join(
    [f"==((\s|\n)*){keyword}((\s|\n)*)==" for keyword in KEYWORDS]
)  # References
RE_REMOVE_H2PLUS = re.compile("={3,} *(.*?) *={3,}")  # Heading 2+
RE_MULTIPLE_SPACES = re.compile("\n{2,}")  # As many spaces as possible
RE_NON_MEANINGFUL_APOSTROPHE = re.compile(
    "'{2,}|([ |,|\.,\s\n])'+|([^s])'+([ |,|\.,\s\n])"
)  # Not apostrophe
TO_REMOVE = ['"', "\*", "#"]
RE_TO_REMOVE = re.compile("|".join([f"[{i}]" for i in TO_REMOVE]))  # Values to remove
RE_SPLIT_SUMMARY = re.compile("== *(.*?) *==")  # split sections


# =============================================================================
# Functions
# =============================================================================


def grouper(iterable, batch_size=10000, fillvalue=None):
    "Group forloop into iterable"
    args = [iter(iterable)] * batch_size
    return zip_longest(fillvalue=fillvalue, *args)


def get_namespace(tag):
    """Returns the namespace of tag."""
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
    return namespace


def my_extract_pages(f, filter_namespaces: list = ["0"]):
    """
    Extract pages from MediaWiki database dump.

    Namespaces and meanings: https://en.wikipedia.org/wiki/Wikipedia:Namespace

    Returns
    -------
    pages : iterable over (str, str)
        Generates (title, content) pairs.
    filter_namespaces: list
        list of namespaces that we care about.
    """

    elems = (elem for _, elem in iterparse(f, events=("end",)))

    # We can't rely on the namespace for database dumps, since it's changed
    # it every time a small modification to the format is made. So, determine
    # those from the first element we find, which will be part of the metadata,
    # and construct element paths.
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping
    pageid_path = "./{%(ns)s}id" % ns_mapping

    print(namespace)
    for elem in elems:
        if elem.tag == page_tag:

            title = elem.find(title_path).text

            text = elem.find(text_path).text

            ns = elem.find(ns_path).text
            if filter_namespaces and ns not in filter_namespaces:
                text = None

            pageid = elem.find(pageid_path).text

            yield title, text or "", pageid  # empty page will yield None

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()


def deal_with_sections(text: str) -> list:
    """
    Cleans up text and splits it into sections.

    Parameters
    ----------
    text : str
        Raw text to clean up.

    Returns
    -------
    list
        Cleaned up text in alternating order of section and text.
        ["Summary", "Text summary", "Section1", "Text Section1",]

    """
    text = re.split(REMOVE_REF, text)[0]  # Remove referemnces
    text = re.sub(RE_REMOVE_H2PLUS, r"\1" + ".", text)  # remove headings H2+
    text = re.sub(
        RE_MULTIPLE_SPACES, r"\n\n", text
    )  # Make multispaces into single space
    text = re.sub(
        RE_NON_MEANINGFUL_APOSTROPHE, r"\1\2\3", text
    )  # Remove non meaningful apostrophes
    text = re.sub(RE_TO_REMOVE, "", text)  # Remove unwanted characters
    text = (
        " " + text
    )  # To avoid errors where there might not be a summary, we will add a single blank space
    text = re.split(RE_SPLIT_SUMMARY, text)  # Split text

    # TODO: check if it is better to remove this before
    text = [remove_markup(section) for section in text]
    text = ["Summary_Target"] + text
    return text


def format_sql_sub_args(section_level_output, article_level_output):
    """Format sub args into SQL input."""
    section_level_output = [
        {
            "pageid": section_level_output[0],
            "section_titles": section_level_output[1],
            "summary": section_level_output[2],
            "body_sections": section_level_output[3],
            "section_word_count": section_level_output[4],
        }
    ]

    article_level_output = [
        {
            "pageid": article_level_output[0],
            "title": article_level_output[1],
            "summary_word_count": article_level_output[2],
            "body_word_count": article_level_output[3],
        }
    ]
    return section_level_output, article_level_output


def my_process_article(queue_read, queue_sql):
    """Perform article processing steps."""

    args = queue_read.get()
    while args is not None:

        # Initialise output lists
        section_level_output_list = []
        article_level_output_list = []

        for sub_arg in args:
            # sub_arg = args[1]
            text, title, pageid = sub_arg

            # print(text)
            # Preprocessing
            text = filter_wiki(text)
            text = deal_with_sections(text)

            # Prepare output
            section_titles_list, section_texts_list = text[::2], text[1::2]

            # Separate summary from text
            section_titles_list = section_titles_list[1:]
            summary = section_texts_list[0]
            body_sections_lists = section_texts_list[1:]

            section_n = len(section_titles_list)

            section_word_count_list = [
                len(nltk.word_tokenize(sect)) for sect in section_texts_list
            ]

            summary_length = section_word_count_list[0]
            body_word_count = sum(section_word_count_list) - summary_length

            section_level_output = (
                pageid,
                pickle.dumps(section_titles_list),
                summary,
                pickle.dumps(body_sections_lists),
                pickle.dumps(section_word_count_list[1:]),
            )
            article_level_output = (
                pageid,
                title,
                summary_length,
                body_word_count,
            )

            # Format outputs and append to lists
            section_level_output, article_level_output = format_sql_sub_args(
                section_level_output, article_level_output
            )
            section_level_output_list += section_level_output
            article_level_output_list += article_level_output

        # yield
        sql_args = (section_level_output_list, article_level_output_list)
        queue_sql.put(sql_args)

        args = queue_read.get()


class MyWikiCorpus:
    def __init__(
        self,
        fname,
        queue_read,
        queue_sql,
        processes=N_PROCESSES,
        filter_namespaces=("0",),
    ):
        self.fname = fname
        self.queue_read = queue_read
        self.queue_sql = queue_sql
        self.processes = processes
        self.filter_namespaces = filter_namespaces

    def my_get_texts(self):
        """Yield Processed wikipedia articles into sql queue."""

        texts_generator = (
            (text, title, pageid)
            for title, text, pageid in my_extract_pages(
                bz2.BZ2File(self.fname),
                self.filter_namespaces,
            )
        )
        # Create processes for text preprocessing
        text_processes = []
        for i in range(self.processes):
            p = Process(
                target=my_process_article, args=(self.queue_read, self.queue_sql)
            )
            text_processes.append(p)
            p.start()

        # Create SQL processes
        sql_process = Process(target=create_wiki_data_base, args=(self.queue_sql,))
        sql_process.start()

        # put data into the read queue
        count_articles = 0
        with tqdm(total=100) as pbar:
            for args in grouper(texts_generator, batch_size=BATCH_SIZE):
                self.queue_read.put(args)
                count_articles += BATCH_SIZE
                pbar.update(BATCH_SIZE)
                if count_articles % (BATCH_SIZE * 1) == 0:
                    print(f"{count_articles} articles processed")

        for _ in range(self.processes):
            self.queue_read.put(None)

        # Join processes
        for p in text_processes:
            # p.terminate()
            p.join()

        queue_sql.put(None)
        # sql_process.terminate()
        sql_process.join()


if __name__ == "__main__":
    start_time = time.time()
    in_f = str(RAW_WIKIPEDIA_CORPUS)

    queue_read = Queue(maxsize=READ_QUE_SIZE)
    queue_sql = Queue(maxsize=SQL_QUE_SIZE)

    wiki_corpus = MyWikiCorpus(
        in_f,
        queue_read=queue_read,
        queue_sql=queue_sql,
    )
    wiki_corpus.my_get_texts()
    end_time = time.time()

    print("finished in: ", str(timedelta(seconds=end_time - start_time)))

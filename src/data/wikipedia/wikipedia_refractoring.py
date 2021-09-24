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
import re
import bz2

from gensim.corpora.wikicorpus import (
    iterparse,
)
from gensim.corpora.wikicorpus import remove_markup
from gensim.corpora.wikicorpus import filter_wiki


# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import DECOMPRESSED_WIKIPEDIA_DUMP, RAW_WIKIPEDIA_CORPUS


KEYWORDS = [
    "notes",
    "sources",
    "primary sources",
    "secondary sources",
    "further reading",
    "external links",
    "see also",
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


# =============================================================================
# Complete
# =============================================================================
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
    text = ["Summary"] + text
    return text


# =============================================================================
# Working on
# =============================================================================


class MyWikiCorpus:
    def __init__(
        self,
        fname,
        processes=None,
        filter_namespaces=("0",),
    ):
        self.fname = fname
        self.processes = processes
        self.filter_namespaces = filter_namespaces


self = MyWikiCorpus(fname=RAW_WIKIPEDIA_CORPUS)


def my_process_article(text, title, pageid):
    text = filter_wiki(text)
    return text, title, pageid


def my_get_texts(self):

    articles, articles_all = 0, 0
    positions, positions_all = 0, 0
    texts_generator = (
        (text, title, pageid)
        for title, text, pageid in my_extract_pages(
            bz2.BZ2File(self.fname),
            self.filter_namespaces,
        )
    )

    for text, title, pageid in texts_generator:
        if len(text) > 400:

            text, _, _ = my_process_article(text, title, pageid)
            text = deal_with_sections(text)
            section_titles, section_texts = text[::2], text[1::2]
            break


# TODO: Deal with #REDIRECT, Apparently redirects are filtered in short articles
# TODO: Add word count
# TODO: segment into sections
# TODO: Multiprocess
# TODO: make into SQL database
# sql_args = my_process_article

# sql_process = Process(target=create_wiki_data_base, args=(self.queue_sql,))
# sql_process.start()

# # Put text read into queue
# test_list = []
# for text in texts_generator:
#     self.queue_read.put(text)

# for _ in range(self.processes):
#     self.queue_read.put(None)
# sql_process.put(None)

# # Join processes
# for p in text_processes:
#     p.join()
# sql_process.join()
# sql_process.join()

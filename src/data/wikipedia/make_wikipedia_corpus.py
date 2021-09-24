# =============================================================================
# Imports
# =============================================================================

import os
import sys
from pathlib import Path

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from multiprocessing import Process, Queue
from itertools import zip_longest

import bz2
import re
from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import (
    filter_wiki,
    iterparse,
)


from src.data.data_statics import RAW_WIKIPEDIA_CORPUS, RAW_DATA_PATH, SQL_WIKI_DUMP
from src.data.wikipedia.wiki_data_base import create_wiki_data_base

# =============================================================================
# Multiprocessing constants
# =============================================================================
BATCH_SIZE = 1000
READ_QUE_SIZE = 1000
SQL_QUE_SIZE = 100

# =============================================================================
# RE compiled
# =============================================================================
PUNCT = "[.,?!]"
TO_REMOVE = ["'", '"', "=", "\*", "#"]
TO_REMOVE_RE = "|".join([f"[{i}]" for i in TO_REMOVE])
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
)

RE_SPLIT_SUMMARY = re.compile("==((.|\n)*?)==".encode("utf-8"))

RE_REMOVE_REF = re.compile(REMOVE_REF.encode("utf-8"))
RE_TO_REMOVE = re.compile(TO_REMOVE_RE.encode("utf-8"))
RE_TO_PARAGRAPH = re.compile("[\s]{2,}".encode("utf-8"))
RE_TO_SPACE = re.compile(" {2,}".encode("utf-8"))
RE_TO_PUNCT_SPACE = [
    re.compile(f"([ ])({PUNCT})".encode("utf-8")),
    r"\2".encode("utf-8"),
]

# =============================================================================
# Functions to modify WikiCorpus class
# =============================================================================
def my_tokenize(content):
    # override original method in wikicorpus.py
    return content.encode("utf-8")


def my_process_article(queue_read, queue_sql):

    args = queue_read.get()
    while args is not None:
        sql_args = []
        for single_arg in args:
            # override original method in wikicorpus.py
            text, lemmatize, title, pageid = single_arg
            text = filter_wiki(text)

            tokens = my_tokenize(text)
            summary, body, n_characters_summary, n_characters_body = split_summary_body(
                tokens
            )

            sql_single_arg = (
                pageid,
                title,
                summary,
                body,
                n_characters_summary,
                n_characters_body,
            )
            sql_args.append(sql_single_arg)

        queue_sql.put(sql_args)
        # return summary, body, n_characters_summary, n_characters_body, title, pageid
        args = queue_read.get()
        print(
            f"Process {os.getpid()} title: {title}  n_characters_summary: {n_characters_summary}"
        )


def get_namespace(tag):
    """Returns the namespace of tag."""
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
    return namespace


def my_extract_pages(f, filter_namespaces=False):
    """
    Extract pages from MediaWiki database dump.
    Returns
    -------
    pages : iterable over (str, str)
        Generates (title, content) pairs.
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
            # if filter_namespaces and ns not in filter_namespaces:
            #     text = None

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


def my_final_text_cleanup(text):
    text = text.strip()
    text = re.sub(RE_TO_REMOVE, "".encode("utf-8"), text)
    text = re.sub(RE_TO_PARAGRAPH, "\n".encode("utf-8"), text)
    text = re.sub(RE_TO_SPACE, " ".encode("utf-8"), text)
    text = re.sub(RE_TO_PUNCT_SPACE[0], RE_TO_PUNCT_SPACE[1], text)
    return text


def split_summary_body(tokens):
    """Splits the wikipedia article between summary and body."""

    # joint_text = b" ".join(tokens)
    joint_text = tokens

    # Split between summary and body
    split_text = re.split(RE_SPLIT_SUMMARY, joint_text, 1)
    summary = split_text[0]
    body = split_text[-1]

    # Remove final sections such as external links, sources and notes
    body = re.split(RE_REMOVE_REF, body, 1)[0]

    # Clean up text characters
    summary = my_final_text_cleanup(summary)
    body = my_final_text_cleanup(body)

    # Get character length
    n_characters_summary = len(summary)
    n_characters_body = len(body)

    return summary, body, n_characters_summary, n_characters_body


def grouper(iterable, batch_size=10000, fillvalue=None):
    "Group forloop into iterable"
    args = [iter(iterable)] * batch_size
    return zip_longest(fillvalue=fillvalue, *args)


class MyWikiCorpus(WikiCorpus):
    def __init__(
        self,
        fname,
        queue_read,
        queue_sql,
        processes=None,
        lemmatize=False,
        dictionary={},
        filter_namespaces=("0",),
        article_min_tokens=0,
        token_min_len=0,
        token_max_len=10e10,
    ):
        WikiCorpus.__init__(
            self,
            fname,
            processes,
            lemmatize,
            dictionary,
            filter_namespaces,
        )
        self.queue_read = queue_read
        self.queue_sql = queue_sql

    def my_get_texts(self, metadata=True):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = (
            (text, self.lemmatize, title, pageid)
            for title, text, pageid in my_extract_pages(
                bz2.BZ2File(self.fname), self.filter_namespaces
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

        sql_process = Process(target=create_wiki_data_base, args=(self.queue_sql,))
        sql_process.start()

        # Put text read into queue
        test_list = []
        for text in grouper(texts, batch_size=BATCH_SIZE):
            self.queue_read.put(text)

        for _ in range(self.processes):
            self.queue_read.put(None)
        sql_process.put(None)

        # Join processes
        for p in text_processes:
            p.join()
        sql_process.join()


def make_corpus(in_f, out_f, queue_read, queue_sql):
    """Convert Wikipedia xml dump file to text corpus"""

    self = MyWikiCorpus(
        in_f,
        lemmatize=False,
        processes=16,
        queue_read=queue_read,
        queue_sql=queue_sql,
    )
    self.my_get_texts()


# if __name__ == "__main__":
#     in_f = str(RAW_WIKIPEDIA_CORPUS)
#     out_f = str(SQL_WIKI_DUMP)

#     queue_read = Queue(maxsize=READ_QUE_SIZE)
#     queue_sql = Queue(maxsize=SQL_QUE_SIZE)

#     make_corpus(in_f, out_f, queue_read, queue_sql)

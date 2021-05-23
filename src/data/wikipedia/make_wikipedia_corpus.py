# =============================================================================
# Imports
# =============================================================================
import multiprocessing
import bz2
import re
from gensim.corpora import WikiCorpus
from gensim.corpora.wikicorpus import (
    filter_wiki,
    iterparse,
    ARTICLE_MIN_WORDS,
    IGNORED_NAMESPACES,
)
from gensim import utils


from src.data.data_statics import (
    RAW_WIKIPEDIA_CORPUS,
    RAW_DATA_PATH,
)


# =============================================================================
# Overide tokenization
# =============================================================================
def my_tokenize(content):
    # override original method in wikicorpus.py
    return [
        token.group().encode("utf8")
        for token in re.finditer(r"[^.,?!\s\n]+|[.,?!\n]", content)
        if len(token.group()) <= 15 and not token.group().startswith("_")
    ]


def my_process_article(args):
    # override original method in wikicorpus.py
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    result = my_tokenize(text)
    return result, title, pageid


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


def my_final_text_cleanup(text):
    text = text.strip()
    text = re.sub(RE_TO_REMOVE, "".encode("utf-8"), text)
    text = re.sub(RE_TO_PARAGRAPH, "\n".encode("utf-8"), text)
    text = re.sub(RE_TO_SPACE, " ".encode("utf-8"), text)
    text = re.sub(RE_TO_PUNCT_SPACE[0], RE_TO_PUNCT_SPACE[1], text)
    return text


def split_summary_body(tokens):
    """Splits the wikipedia article between summary and body."""

    joint_text = b" ".join(tokens)

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


class MyWikiCorpus(WikiCorpus):
    def __init__(
        self,
        fname,
        processes=None,
        lemmatize=False,
        dictionary={},
        filter_namespaces=("0",),
    ):
        WikiCorpus.__init__(
            self,
            fname,
            processes,
            lemmatize,
            dictionary,
            filter_namespaces,
        )

    def my_get_texts(self, metadata=True):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = (
            (text, self.lemmatize, title, pageid)
            for title, text, pageid in my_extract_pages(
                bz2.BZ2File(self.fname), self.filter_namespaces
            )
        )
        pool = multiprocessing.Pool(self.processes)
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(
                my_process_article, group
            ):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
            if len(tokens) < ARTICLE_MIN_WORDS or any(
                title.startswith(ignore + ":") for ignore in IGNORED_NAMESPACES
            ):
                continue
            articles += 1
            positions += len(tokens)

            summary, body, n_characters_summary, n_characters_body = split_summary_body(
                tokens
            )

            yield summary, body, n_characters_summary, n_characters_body, pageid, title

        pool.terminate()

        print(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles,
            positions,
            articles_all,
            positions_all,
            ARTICLE_MIN_WORDS,
        )
        self.length = articles  # cache corpus length


def make_corpus(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""

    wiki = MyWikiCorpus(in_f, lemmatize=False, processes=16)

    i = 0
    for (
        summary,
        body,
        n_characters_summary,
        n_characters_body,
        pageid,
        title,
    ) in wiki.my_get_texts():

        print(pageid)


# if __name__ == "__main__":

#     in_f = str(RAW_WIKIPEDIA_CORPUS)
#     out_f = str(DUMP_WIKIPEDIA_CORPUS)
#     make_corpus(in_f, out_f)


in_f = str(RAW_WIKIPEDIA_CORPUS)

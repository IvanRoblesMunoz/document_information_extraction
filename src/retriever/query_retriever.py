#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:28:30 2021

@author: roblesi
"""

# =============================================================================
# Imports
# =============================================================================
import pickle
import sqlite3

import numpy as np

import faiss

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from src.retriever.run_make_faiss_indeces import (
    create_faiss_document_store,
    initialise_faiss_retriever,
)
from src.data.wikipedia.wiki_data_base import retrive_observations_from_ids

# =============================================================================
# Statics
# =============================================================================

from src.retriever.retriever_statics import (
    BM25_PASSAGE_WEIGHT,
    BM25_TITLE_WEIGHT,
    BM25_LIMIT_SEARCH,
)
from src.data.data_statics import SQL_WIKI_DUMP

FAISS_SEMANTIC_SIMIL_DIM = 768
ENGLISH_STOPWORDS = set(stopwords.words("english"))
ENGLISH_PUNCTUATION = set(string.punctuation)
PORTER_STEMMER = PorterStemmer()

# =============================================================================
# BM25 retriever
# =============================================================================


def clean_up_search_query(search_query):
    """Clean up query so to optimise search."""
    search_query = set(word_tokenize(search_query))
    search_query = search_query - ENGLISH_STOPWORDS
    search_query = search_query - ENGLISH_PUNCTUATION
    search_query = [PORTER_STEMMER.stem(token) for token in search_query]
    return search_query


def produce_formated_bm25_search_query(search_query):
    """Format query so that its compatible with search."""
    search_query = clean_up_search_query(search_query)
    search_query = " OR ".join(search_query)
    return search_query


def retrieve_using_bm25(
    query_search,
    cur,
    passage_weight=BM25_PASSAGE_WEIGHT,
    title_weight=BM25_TITLE_WEIGHT,
    limit_articles=BM25_LIMIT_SEARCH,
):
    formated_query_search = produce_formated_bm25_search_query(query_search)
    formated_query_search = (
        f"(passage: {formated_query_search}) AND (title: {formated_query_search})"
    )

    template_query_bm25 = f"""
        SELECT *, 
               bm25(bm25_wiki_articles,
                    {passage_weight},
                    {title_weight}
                )
        FROM bm25_wiki_articles
        WHERE bm25_wiki_articles MATCH "{formated_query_search}"
        ORDER BY bm25(bm25_wiki_articles,{passage_weight},{title_weight})
        LIMIT {limit_articles}
    """
    res = cur.execute(template_query_bm25).fetchall()
    return res


class WikiODQARetriever:
    def __init__(self):
        self.document_store = create_faiss_document_store()
        self.retriever = initialise_faiss_retriever(self.document_store)
        self.decoded_articles = None
        self.decoded_embeddings = None
        self.embedded_query = None

    def _format_output(self, all_faiss_articles):
        decoded_articles = []
        decoded_embeddings = []

        for article in all_faiss_articles:
            pageid, title, embeddings, body_sections = article
            embeddings = pickle.loads(embeddings)
            body_sections = pickle.loads(body_sections)

            for passage, passage_embedding in zip(body_sections, embeddings):
                decoded_articles.append([pageid, title, passage.content])
                decoded_embeddings.append(passage_embedding)
        return decoded_articles, decoded_embeddings

    def _make_faiss_index(self):
        # self.faiss_index = faiss.IndexFlatL2(FAISS_SEMANTIC_SIMIL_DIM)

        self.faiss_index = faiss.IndexFlatIP(FAISS_SEMANTIC_SIMIL_DIM)
        self.faiss_index.add(np.stack(self.decoded_embeddings))

    def _extract_queries_faiss(self, query_search, out_f=SQL_WIKI_DUMP, verbose=True):

        db = sqlite3.connect(out_f)
        cur = db.cursor()

        # if verbose:
        #     print("Retrieving articles using BM25 ...")
        relevant_bm25_articles = retrieve_using_bm25(query_search, cur)

        pageids, _, _, _ = zip(*relevant_bm25_articles)
        pageids = list(set(pageids))

        faiss_generator = retrive_observations_from_ids(
            pageids,
            out_f=out_f,
            table="faiss_embedding_store",
        )

        all_faiss_articles = [article_batch for article_batch in faiss_generator]
        decoded_articles, decoded_embeddings = self._format_output(all_faiss_articles)
        return decoded_articles, decoded_embeddings

    def retrieve_passage_from_query(
        self, query_search: str, top_n_answers: int = 20
    ) -> list:
        self.decoded_articles, self.decoded_embeddings = self._extract_queries_faiss(
            query_search
        )
        self.embedded_query = self.retriever.embed_queries([query_search])

        self._make_faiss_index()

        response = self.faiss_index.search(self.embedded_query, top_n_answers)

        all_responses = []
        for rank, (top_idx, top_similarity) in enumerate(
            zip(response[1][0], response[0][0])
        ):
            response_dict = dict()
            pageid, title, text = self.decoded_articles[top_idx]
            response_dict["pageid"] = pageid
            response_dict["title"] = title
            response_dict["text"] = text
            response_dict["similarity"] = top_similarity
            response_dict["rank"] = rank

            all_responses.append(response_dict)

        return all_responses

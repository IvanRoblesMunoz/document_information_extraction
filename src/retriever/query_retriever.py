#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:28:30 2021

@author: roblesi
"""

# =============================================================================
# Imports
# =============================================================================
import faiss

from transformers import AutoTokenizer, DPRQuestionEncoder


from sentence_transformers import SentenceTransformer
from haystack.preprocessor import PreProcessor

from src.retriever.retriever_input_articles import processed_document_generator

# =============================================================================
# Statics
# =============================================================================
from src.retriever.retriever_statics import (
    FAISS_MAX_PASSAGE_TOKEN_LEN,
    FAISS_DB_DATA_PATH,
    FAISS_EMB_DATA_PATH,
    FAISS_QUERY_EMBEDDING_MODEL,
    FAISS_PASSAGE_EMBEDDING_MODEL,
    FAISS_CONFIG_DATA_PATH,
)

FAIIS_SEMANTIC_SIMIL_SEARCH_BEST = "all-mpnet-base-v2"
FAISS_SEMANTIC_SIMIL_DIM = 768
# Max Sequence Length:	384
# Dimensions:	768

# =============================================================================
# Functions
# =============================================================================

faiss_pre_processor = PreProcessor(
    split_length=FAISS_MAX_PASSAGE_TOKEN_LEN,
)
faiss_generator = processed_document_generator(
    storage_method="faiss",
    preprocessor=faiss_pre_processor,
    **{"n_sample_articles": 1000},
)

for batch in faiss_generator:
    pass


tokenizer_passage = AutoTokenizer.from_pretrained(FAISS_PASSAGE_EMBEDDING_MODEL)
model_passage = DPRQuestionEncoder.from_pretrained(FAISS_PASSAGE_EMBEDDING_MODEL)

batch[0].content  # ["content"]
batch[0].meta["page_id"]
len(batch)

print("loading model...")
model = SentenceTransformer(FAIIS_SEMANTIC_SIMIL_SEARCH_BEST)

print("encoding passages...")
batch_vectors = model.encode([i.content for i in batch])

print("make index...")
index = faiss.IndexFlatL2(FAISS_SEMANTIC_SIMIL_DIM)

print("adding indeces...")
index.add(batch_vectors)

query = ["What is autism?"]
query_vectors = model.encode(query)

response = index.search(query_vectors, 5)


response_doc = [(batch[i].content, batch[i].meta["title"]) for i in response[1][0]]

help(index.search)


# =============================================================================
# Query search
# =============================================================================
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


ENGLISH_STOPWORDS = set(stopwords.words("english"))
ENGLISH_PUNCTUATION = set(string.punctuation)
PORTER_STEMMER = PorterStemmer()
BM25_PASSAGE_WEIGHT = 10
BM25_TITLE_WEIGHT = 10
BM25_LIMIT_SEARCH = 1000

search_query = "what is the philosophy behind Anarchism?"


def clean_up_search_query(search_query):
    search_query = set(word_tokenize(search_query))
    search_query = search_query - ENGLISH_STOPWORDS
    search_query = search_query - ENGLISH_PUNCTUATION
    search_query = [PORTER_STEMMER.stem(token) for token in search_query]
    return search_query


def produce_formated_bm25_search_query(search_query):
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
    formated_query_search = clean_up_search_query(search_query)
    formated_query_search = f"(passage: {query_search}) AND (title: {query_search})"

    template_query_bm25 = f"""
        SELECT *, bm25(bm25_wiki_articles,{passage_weight},{title_weight})
        FROM bm25_wiki_articles
        WHERE bm25_wiki_articles MATCH "{formated_query_search}"
        ORDER BY bm25(bm25_wiki_articles,{passage_weight},{title_weight})
        LIMIT {limit_articles}
    """
    res = cur.execute(template_query_bm25).fetchall()
    return res

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:15:34 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import re
from bert_score.utils import get_tokenizer
from src.data.data_statics import MODEL_TYPE

# =============================================================================
# Statics
# =============================================================================

RE_SENT_TOKENIZER_BYTES = re.compile(
    b"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(?:\s|[A-Z].*)"
)
RE_SENT_TOKENIZER_STRING = re.compile(
    "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(?:\s|[A-Z].*)"
)

# =============================================================================
# Functions
# =============================================================================


def tokenize_sentence(sentence: str) -> list:
    if type(sentence) == str:
        re_tokenizer = RE_SENT_TOKENIZER_STRING
    elif type(sentence) == bytes:
        re_tokenizer = RE_SENT_TOKENIZER_STRING
    else:
        raise TypeError("type must be string or bytes")

    return re.split(re_tokenizer, sentence)


class ModelTokenizer:
    def __init__(self, model_type=MODEL_TYPE):
        self.tokenizer = get_tokenizer(model_type)

    def tokenize(self, text):
        tokenized_text = self.tokenizer(text)
        return tokenized_text

    def count_tokens(self, text):
        tokenized_text = self.tokenizer(text)
        n_tokens = len(tokenized_text["input_ids"])
        return n_tokens

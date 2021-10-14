#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 19:46:34 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    # BertTokenizerFast as BertTokenizer,
    # BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc


from src.summarisation.data_query_summariser import get_article_ids
from src.data.wikipedia.wiki_data_base import retrive_observations_from_ids

# =============================================================================
# Statics
# =============================================================================
RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)


BART_MODEL_NAME = "facebook/bart-large-cnn"
BERT_MODEL_NAME = "bert-base-cased"

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", TORCH_DEVICE)
language = "english"

tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME).to(TORCH_DEVICE)

print(model.config)
ENCODER_MAX_LENGTH = 1024
DECODER_MAX_LENGTH = 64

# =============================================================================
# Data
# =============================================================================
all_ids = get_article_ids(
    random_state=RANDOM_SEED, n_sample_texts=100, max_tokens_body=1022
)

article_ids_train = all_ids[:97]
article_ids_test = all_ids[97:]


def decode_row(article):
    summary = article[2]
    body = "".join(pickle.loads(article[3]))
    return summary, body


def convert_ids_to_features(example_ids: list) -> dict:
    """Retrieve ids and convert them to document and target dict."""

    example_batch = retrive_observations_from_ids(example_ids)

    input_ = []
    target_ = []
    for article in example_batch:
        summary, body = decode_row(article)
        input_.append(body)
        target_.append(summary)

    return {"document": input_, "summary": target_}


def encode(tokenizer, data, max_length):
    encoding = tokenizer.encode_plus(
        data,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
        is_split_into_words=False,
    )
    return encoding


train_data = convert_ids_to_features([article_ids_train[1]])
sample_summary = encode(tokenizer, train_data["summary"][0], DECODER_MAX_LENGTH)

print(sample_summary.keys())
print(sample_summary["input_ids"].shape)
print(sample_summary["attention_mask"].shape)

print(tokenizer.convert_ids_to_tokens(sample_summary["input_ids"].squeeze())[:20])


class SummaryDataset(Dataset):
    def __init__(
        self,
        article_ids: list,
        tokenizer: AutoTokenizer,
        max_input_len: int = ENCODER_MAX_LENGTH,
        max_output_len: int = DECODER_MAX_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.article_ids = article_ids
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, index: int):
        # index= 1
        data_row = convert_ids_to_features([self.article_ids[index]])

        body_encoding = encode(
            self.tokenizer, data_row["document"][0], max_length=self.max_input_len
        )
        summary_encoding = encode(
            self.tokenizer, data_row["summary"][0], max_length=self.max_output_len
        )

        return dict(
            input_ids=body_encoding["input_ids"].to(TORCH_DEVICE),  # .flatten(),
            attention_mask=body_encoding["attention_mask"].to(
                TORCH_DEVICE
            ),  # .flatten(),
            targets_ids=summary_encoding["input_ids"].to(TORCH_DEVICE),  # .flatten(),
            target_mask=summary_encoding["attention_mask"].to(
                TORCH_DEVICE
            ),  # .flatten(),
        )


dataset = SummaryDataset(article_ids=article_ids_train, tokenizer=tokenizer)

sample_item = dataset[0]
sample_item.keys()

sample_item["input_ids"].shape


output = model(sample_item["input_ids"], sample_item["attention_mask"])
type(output)

# TODO: Figure out how to decode this
for i in output.keys():
    try:
        print(i, output[i].shape)
    except AttributeError:
        print(i, len(output[i]))

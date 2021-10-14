#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 19:46:34 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from transformers import BartTokenizer, BartForConditionalGeneration

# =============================================================================
# Stuff
# =============================================================================
from src.data.data_statics import (
    MIN_SEMANTIC_SIMILARITY,
    MIN_NOVELTY,
    MAX_NOVELTY,
    MAX_TOKENS_BODY,
)
import pandas as pd
from src.data.wikipedia.wiki_data_base import (
    retrieve_query,
    retrive_observations_from_ids,
)

MODEL_NAME = "facebook/bart-large-cnn"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", TORCH_DEVICE)

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(TORCH_DEVICE)

RANDOM_SEED = 0
N_SAMPLE_TEXTS = 1000

QUERY_SUITABLE_ARTICLES = f"""
SELECT ar.*,
       nv.novelty_tokens,
       nv.novelty_bigrams,
       nv.novelty_trigrams,
       cs.semantic_similarity
       
FROM article_level_info ar
INNER JOIN wiki_article_novelty nv
    ON ar.pageid = nv.pageid
INNER JOIN wiki_article_cosine_similarity cs
    ON ar.pageid = cs.pageid
WHERE cs.semantic_similarity>={MIN_SEMANTIC_SIMILARITY}
    AND nv.novelty_tokens<={MAX_NOVELTY}
    AND nv.novelty_tokens>={MIN_NOVELTY}
    AND ar.body_word_count<={MAX_TOKENS_BODY}
"""
import pickle


characterisation_df = pd.DataFrame(
    retrieve_query(QUERY_SUITABLE_ARTICLES),
    columns=[
        "pageid",
        "title",
        "summary_word_count",
        "body_word_count",
        "novelty_tokens",
        "novelty_bigrams",
        "novelty_trigrams",
        "semantic_similarity",
    ],
)


pageids_to_evaluate = list(
    characterisation_df["pageid"].sample(n=N_SAMPLE_TEXTS, random_state=RANDOM_SEED)
)
ARTICLE_GENERATOR = retrive_observations_from_ids(pageids_to_evaluate)


class DataLoader(Dataset):
    """Wikipedia data loader."""

    def __init__(
        self,
        tokenizer,
        dataset,
        retriever=retrive_observations_from_ids,
        output_length=124,  # default BART LARGE CNN
        input_length=1024,  # default BART LARGE CNN
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.retriever = retriever
        self.output_length = output_length
        self.input_length = input_length

    def __len__(self):
        return self.dataset.shape[0]

    def decode_row(self, article):
        summary = article[2]
        body = "".join(pickle.loads(article[3]))
        return summary, body

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        input_ = []
        target_ = []
        for article in example_batch:
            summary, body = self.decode_row(article)
            input_.append(body)
            target_.append(summary)

        source = self.tokenizer.batch_encode_plus(
            input_,
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer.batch_encode_plus(
            target_,
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return source, targets

    def __getitem__(self, index):
        example_batch = list(self.retriever(self.dataset[index]))
        source, targets = self.convert_to_features(example_batch)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }


self = DataLoader(
    tokenizer=tokenizer,
    dataset=pageids_to_evaluate,
)
index = slice(1, 2)
sample_data = self.__getitem__(slice(1, 2))

sample_data["source"]
# =============================================================================
# Trainer with lightining module
# =============================================================================
from torch.utils.tensorboard import SummaryWriter

sample_data2 = (sample_data["source_ids"], sample_data["source_mask"])

sample_data["source_ids"].shape
sample_data["source_mask"].shape

writer = SummaryWriter("runs")
writer.add_graph(model, sample_data["source_ids"])
writer.close()


# self = DataLoader(
#     tokenizer=tokenizer,
#     dataset=pageids_to_evaluate,
# )
hparams = {"n_train": 100, "n_val": 100, "n_test": 100}


class SummariserFineTuner(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer


self = SummariserFineTuner(model=model, tokenizer=tokenizer)

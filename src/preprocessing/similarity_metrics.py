#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 00:02:37 2021

@author: ivanr
"""

# =============================================================================
# For testing
# =============================================================================

# import sys
# import os

# # os.chdir("/home/ivanr/git/document_information_extraction")
# # sys.path.insert(1, "/home/ivanr/git/document_information_extraction")

# # # os.chdir("/home/ivanr/git/document_information_extraction/")
# # sys.path.append("/home/ivanr/git/document_information_extraction")
# # sys.path.append("/home/ivanr/git/document_information_extraction/src")

# # import src.data.wikipedia.wiki_data_base as wdb

# from src.data.wikipedia.wiki_data_base import (
#     retrive_observations_from_ids,
#     retrive_suitable_column_ids,
# )


# ids = retrive_suitable_column_ids(limit_ids=150)
# # ids = retrive_suitable_column_ids(limit_ids=None)
# # len(ids)
# relevant_obs = retrive_observations_from_ids(ids)
# obs = relevant_obs[0:150]
# # obs2 = relevant_obs[1]

# summary = [i[1].decode("utf-8") for i in obs]
# text = [i[2].decode("utf-8") for i in obs]

# from src.preprocessing.tokenization import tokenize_sentence

# obs[0]

# # Parameters
# refs = [[i] for i in text]
# cands = [tokenize_sentence(i) for i in summary]
# article_ids = [i[0] for i in obs]
import re
import time

# =============================================================================
# Imports
# =============================================================================
from collections import defaultdict
import itertools
from tqdm import tqdm

import torch
from bert_score.utils import (
    get_bert_embedding,
    pad_sequence,
    greedy_cos_idf,
    lang2model,
    model2layers,
    get_tokenizer,
    get_model,
)

from src.data.data_statics import MODEL_TYPE, NUM_LAYERS, ALL_LAYERS, LANGUAGE

# =============================================================================
# BERT Score
# =============================================================================
def pad_batch_stats(sen_batch, stats_dict, device):
    stats = [stats_dict[s] for s in sen_batch]
    emb, idf = zip(*stats)
    emb = [e.to(device) for e in emb]
    idf = [i.to(device) for i in idf]
    lens = [e.size(0) for e in emb]
    emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
    idf_pad = pad_sequence(idf, batch_first=True)

    def length_to_mask(lens):
        lens = torch.tensor(lens, dtype=torch.long)
        max_len = max(lens)
        base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
        return base < lens.unsqueeze(1)

    pad_mask = length_to_mask(lens).to(device)
    return emb_pad, pad_mask, idf_pad


class BERTScore:
    """
    The default model is as shown bellow with the following performances.
    model: allenai/led-base-16384
    best layer:6
    corr:0.7122
    rank:29
    max tokens:16382
    """

    def __init__(
        self,
        lang: str = LANGUAGE,
        model_type: str = MODEL_TYPE,
        num_layers: int = NUM_LAYERS,
        all_layers: bool = ALL_LAYERS,
        rescale_with_baseline: bool = False,
        device: str = None,
        verbose: bool = True,
    ):
        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers
        self.all_layers = all_layers
        self.rescale_with_baseline = rescale_with_baseline
        self.device = device
        self.verbose = verbose

        self._perform_attribute_checks()
        self._load_model_related_attributes()

    def _perform_attribute_checks(self):
        if self.rescale_with_baseline:
            assert (
                self.lang is not None
            ), "Need to specify Language when rescaling with baseline"

    def _load_model_related_attributes(self):

        if self.model_type is None:
            self.lang = self.lang.lower()
            self.model_type = lang2model[self.lang]

        if self.num_layers is None:
            self.num_layers = model2layers[self.model_type]

        # Get model and tokenizer
        self.tokenizer = get_tokenizer(self.model_type)
        self.model = get_model(self.model_type, self.num_layers, self.all_layers)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

        # TODO: check what this is and why it is important
        self._idf_dict = defaultdict(lambda: 1.0)
        self._idf_dict[self.tokenizer.sep_token_id] = 0
        self._idf_dict[self.tokenizer.cls_token_id] = 0

    def _produce_embedings(self, sentences, batch_name, batch_size=1):
        embs = []
        iter_range = range(0, len(sentences), batch_size)
        if self.verbose:
            print(f"computing bert embedding for {batch_name}.")
            iter_range = tqdm(iter_range)
        stats_dict = dict()

        for batch_start in iter_range:

            sen_batch = sentences[batch_start : batch_start + batch_size]
            embs, masks, padded_idf = get_bert_embedding(
                sen_batch,
                self.model,
                self.tokenizer,
                self._idf_dict,
                device=self.device,
                all_layers=self.all_layers,
            )

            embs = embs.cpu()
            masks = masks.cpu()
            padded_idf = padded_idf.cpu()
            for i, sen in enumerate(sen_batch):
                sequence_len = masks[i].sum().item()
                emb = embs[i, :sequence_len]
                idf = padded_idf[i, :sequence_len]
                stats_dict[sen] = (emb, idf)

        return stats_dict

    def make_embedding_dictionary(
        self, refs: list, hyps: list, batch_size_ref: int = 1, batch_size_hyp: int = 20
    ) -> dict:
        """
        Make a dictionary containing sentence embeddings.

        Parameters
        ----------
        refs : list[list[str]]
            Reference sentences.
        hyps : list[list[str]]
            Candidate sentences.
        batch_size_ref : int, optional
            Batch size to use during reference embedding. The default is 1.
        batch_size_hyp : int, optional
            Batch size to use during candidate embedding.. The default is 20.

        Returns
        -------
        dict
            Dictionary containing sentence as keys and embeddings as values.

        """
        refs_flat = list(itertools.chain(*refs))
        hyps_flat = list(itertools.chain(*hyps))

        stats_dict = self._produce_embedings(
            refs_flat, batch_size=batch_size_ref, batch_name="reference sentences"
        )
        stats_dict.update(
            self._produce_embedings(
                hyps_flat, batch_size=batch_size_hyp, batch_name="candidate sentences"
            )
        )
        return stats_dict

    def bert_cos_score_idf(
        self, refs: str, hyps: str, stats_dict: dict, verbose: bool = True
    ):
        """
        Produce final bert score.

        Parameters
        ----------
        refs : str
            Reference sentence.
        hyps : str
            Candidate sentence.
        stats_dic : dict
            Dictionary containg sentence embedings.
        verbose : bool, optional
            Verbose. The default is True.

        Returns
        -------
        preds : TYPE
            array[P, R, F].

        """
        preds = []

        device = next(self.model.parameters()).device

        with torch.no_grad():
            batch_refs = [refs]
            batch_hyps = [hyps]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)

            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, self.all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())

        preds = torch.cat(preds, dim=1 if self.all_layers else 0)
        return preds.numpy()[0]


# import time

# start_time = time.time()
# bert_scorer = BERTScore()
# load_time = time.time()
# stats_dict = bert_scorer.make_embedding_dictionary(refs, cands)
# embedding_time = time.time()
# final_results = []

# for idx, sumry, txt in zip(article_ids, cands, refs):
#     for sentence_idx, sumry_sentence in enumerate(sumry):
#         preds = bert_scorer.bert_cos_score_idf( txt[0],sumry_sentence, stats_dict)
#         final_results.append([idx, sentence_idx]+ list(preds)+[ sumry_sentence] )
# prediction_time = time.time()
# print(
#     f"load time = {load_time-start_time}\nembedding_time = {embedding_time-load_time}\nprediction_time = {prediction_time-embedding_time}"
# )

# =============================================================================
# Compute bert score
# =============================================================================


# refs = refs[0]
# cands= [cands]


# Use this to free up gpu memory
# del self
# del stats_dict
# torch.cuda.empty_cache()

# Available 2697750

# 3.57*60*60*24*3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:36:16 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
from tqdm import tqdm
from pathlib import Path
import random

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))
# import torch
from src.data.wikipedia.wiki_data_base import (
    retrive_suitable_column_ids,
    retrive_observations_from_ids,
)
from src.preprocessing.tokenization import tokenize_sentence
from src.evaluation.similarity_metrics import BERTScore
from src.data.wikipedia.wiki_data_base import (
    create_summary_sentence_similarity_database,
)

# =============================================================================
# Statics
# =============================================================================
BATCH_SIZE = 100
N_SAMPLES = 100000
SEED = 0
# =============================================================================
# Functions
# =============================================================================

bert_scorer = BERTScore()

print("Reading suitable article ids")
ids = retrive_suitable_column_ids()
print(f"{len(ids)} suitable articles found, using {N_SAMPLES}")
ids = random.sample(ids, N_SAMPLES)


for i in tqdm(range(0, len(ids), BATCH_SIZE)):
    batch = retrive_observations_from_ids(ids[i : i + BATCH_SIZE])
    print(f"{i} articles read out of {len(ids)}")

    summary = [i[1].decode("utf-8") for i in batch]
    text = [i[2].decode("utf-8") for i in batch]

    refs = [[i] for i in text]
    cands = [tokenize_sentence(i) for i in summary]
    article_ids = [i[0] for i in batch]
    stats_dict = {}
    stats_dict = bert_scorer.make_embedding_dictionary(refs, cands)

    final_results = []

    for idx, sumry, txt in zip(article_ids, cands, refs):
        for sentence_idx, sumry_sentence in enumerate(sumry):
            preds = bert_scorer.bert_cos_score_idf(txt[0], sumry_sentence, stats_dict)
            final_results.append(
                tuple(
                    [str(idx) + "_" + str(sentence_idx), idx, sentence_idx]
                    + list(preds)
                    + [sumry_sentence]
                )
            )

    create_summary_sentence_similarity_database(final_results)


#         # del bert_scorer
#         # del stats_dict
#         stats_dict = {}

#         torch.cuda.empty_cache()

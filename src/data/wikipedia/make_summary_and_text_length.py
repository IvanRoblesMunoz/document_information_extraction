#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:15:26 2021

@author: ivanr
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
from pathlib import Path

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from multiprocessing import Process, Queue

from src.data.wikipedia.wiki_data_base import (
    retrieve_query,
    retrive_observations_from_ids,
    create_text_length_database,
)
from src.preprocessing.tokenization import tokenize_sentence, ModelTokenizer

# =============================================================================
# Constants
# =============================================================================
READ_QUE_SIZE = 160
PROCESS_QUE_SIZE = 160
SQL_QUE_SIZE = 160
N_PROCESSES = 16
BATCH_SIZE = 2500

# =============================================================================
# Run
# =============================================================================
# Retrieve Ids
print("retrieving ids")
query = """
        SELECT pageid
        FROM wiki_articles
        """

ids = retrieve_query(query)
print("retrieving model tokenizer")
model_tokenizer = ModelTokenizer()

# Retrive batches
def retrieve_data(queue_read, queue_process):
    args = queue_read.get()
    while args is not None:
        ids_retrieve, id_track = args
        batch_observation = retrive_observations_from_ids(ids_retrieve)
        queue_process.put((batch_observation, id_track))
        args = queue_read.get()


def process_data(queue_process, queue_sql):
    args = queue_process.get()

    while args is not None:
        batch_observation, id_track = args

        batch_lengths = []
        for obs in batch_observation:
            page_id = obs[0]
            summary = obs[1].decode("utf-8")
            text = obs[2].decode("utf-8")

            n_sentence_summary = len(tokenize_sentence(summary))
            n_sentence_text = len(tokenize_sentence(text))

            n_model_tokens_summary = model_tokenizer.count_tokens(summary)
            n_model_tokens_text = model_tokenizer.count_tokens(text)

            batch_lengths.append(
                (
                    page_id,
                    n_sentence_summary,
                    n_sentence_text,
                    n_model_tokens_summary,
                    n_model_tokens_text,
                )
            )
        queue_sql.put(batch_lengths)

        if id_track % 1000 == 0:
            print(f"Processed {id_track} articles")

        args = queue_process.get()


if __name__ == "__main__":
    print("Start processes")
    queue_read = Queue(maxsize=READ_QUE_SIZE)
    queue_process = Queue(maxsize=PROCESS_QUE_SIZE)
    queue_sql = Queue(maxsize=SQL_QUE_SIZE)

    process_list = []
    for _ in range(N_PROCESSES):
        p = Process(target=retrieve_data, args=(queue_read, queue_process))
        process_list.append(p)
        p.start()

    for _ in range(N_PROCESSES):
        p = Process(target=process_data, args=(queue_process, queue_sql))
        process_list.append(p)
        p.start()

    p = Process(target=create_text_length_database, args=(queue_sql,))
    process_list.append(p)
    p.start()

    for i in range(0, len(ids), BATCH_SIZE):
        ids_retrieve = ids[i : i + BATCH_SIZE]
        ids_retrieve = [i[0] for i in ids_retrieve]
        queue_read.put((ids_retrieve, i))

    for p in process_list:
        p.put(None)

    for p in process_list:
        p.join()

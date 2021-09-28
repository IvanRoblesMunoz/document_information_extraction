#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 21:44:13 2021

@author: ivanr
"""
# =============================================================================
# Imports
# =============================================================================
import sys
import os
import time
from datetime import timedelta
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from tqdm import tqdm

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from src.characterisation.functions import (
    compute_semantic_similarity,
    prepare_batch_data,
)
from src.data.wikipedia.wiki_data_base import (
    retrive_suitable_strings,
    insert_observations_in_table_mp,
    WikiCosineSimilarity,
    transfer_to_new_db,
    semantic_similarity_data_input_formater,
)

# =============================================================================
# Statics
# =============================================================================

from src.data.data_statics import (
    BATCH_SIZE_SEMANTIC_SIMILARITY,
    TEMP_DB,
    SQL_WIKI_DUMP,
    SEM_SIM_READ_QUE_SIZE,
    SEM_SIM_SQL_QUE_SIZE,
    SEM_SIM_PREP_QUE_SIZE,
    SEM_SIM_N_PROCESSES,
)

# =============================================================================
# Functions
# =============================================================================


def process_batch_prep(queue_read, queue_prep):
    """Preprocess args by summarising body to produce batch."""

    batch = queue_read.get()

    while not batch is None:
        encoding_batch = prepare_batch_data(batch)

        # Yield results
        queue_prep.put((encoding_batch))
        batch = queue_read.get()


def process_batch_semantic_similarity(queue_prep, queue_sql):
    """Process novelty."""

    encoding_batch = queue_prep.get()

    while not encoding_batch is None:
        sql_args = compute_semantic_similarity(encoding_batch)

        # Yield results
        queue_sql.put(sql_args)
        encoding_batch = queue_prep.get()


def main(queue_read, queue_prep, queue_sql, n_processes):
    # Create generator to read data from SQL db
    article_generator = retrive_suitable_strings(
        batchsize=BATCH_SIZE_SEMANTIC_SIMILARITY
    )

    # Create semantic similarity calculator process
    process_sem_sim = Process(
        target=process_batch_semantic_similarity, args=(queue_prep, queue_sql)
    )
    process_sem_sim.start()

    batch_process_list = []
    for p in range(n_processes):
        p = Process(target=process_batch_prep, args=(queue_read, queue_prep))
        batch_process_list.append(p)
        p.start()

    # Create SQL processes
    sql_process = Process(
        target=insert_observations_in_table_mp,
        args=(queue_sql, WikiCosineSimilarity, TEMP_DB),
    )
    sql_process.start()

    # put data into the read queue
    count_articles = 0
    # I know that there are ~1.5M suitable articles but this is not necessarilly correct
    with tqdm(total=1.4e6) as pbar:
        for args in article_generator:
            queue_read.put(args)

            # Track progress
            count_articles += BATCH_SIZE_SEMANTIC_SIMILARITY
            pbar.update(BATCH_SIZE_SEMANTIC_SIMILARITY)
            if count_articles % (BATCH_SIZE_SEMANTIC_SIMILARITY * 1) == 0:
                print(f"{count_articles} articles processed")

    # Terminate
    for _ in batch_process_list:
        queue_read.put(None)
    for p in batch_process_list:
        p.join()

    queue_prep.put(None)
    process_sem_sim.join()

    queue_sql.put(None)
    sql_process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # https://stackoverflow.com/questions/3172929/operationalerror-database-is-locked
    # Apparently SQL lite does not support concurrent operations so we cant
    # read using a generator and import to the same database since two connections
    # will be required to the database. Instead we will create a temporary db
    # # and then copy the content and delete it.

    start_time = time.time()

    # Make semantic similarity db
    queue_read = Queue(maxsize=SEM_SIM_READ_QUE_SIZE)
    queue_prep = Queue(maxsize=SEM_SIM_PREP_QUE_SIZE)
    queue_sql = Queue(maxsize=SEM_SIM_SQL_QUE_SIZE)
    main(queue_read, queue_prep, queue_sql, SEM_SIM_N_PROCESSES)

    # Transfer to main db
    src_query = """
        SELECT *
        FROM wiki_article_cosine_similarity
    """
    transfer_to_new_db(
        src_query,
        src_db=TEMP_DB,
        dest_db=SQL_WIKI_DUMP,
        dest_table=WikiCosineSimilarity,
        batch_formater=semantic_similarity_data_input_formater,
    )

    # Delete temporary db
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    end_time = time.time()
    print("finished in: ", str(timedelta(seconds=end_time - start_time)))

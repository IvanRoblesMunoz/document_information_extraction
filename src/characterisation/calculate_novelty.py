#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 11:27:05 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import os
import sys
from multiprocessing import Process, Queue
from pathlib import Path
import pickle
import time
from datetime import timedelta
from tqdm import tqdm

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))

from src.data.wikipedia.wiki_data_base import (
    retrive_suitable_strings,
    insert_observations_in_table_mp,
    WikiArticleNovelty,
    transfer_to_new_db,
    novelty_data_input_formater,
)
from src.characterisation.functions import n_gram_novelty_calculator

# =============================================================================
# Statics
# =============================================================================

from src.data.data_statics import (
    BATCH_SIZE_NOVELTY,
    NOVELTY_READ_QUE_SIZE,
    NOVELTY_SQL_QUE_SIZE,
    NOVELTY_N_PROCESSES,
    TEMP_DB,
    SQL_WIKI_DUMP,
)

# =============================================================================
# Functions
# =============================================================================


def decode_row(row: list) -> tuple:
    """
    Decodes row.

    Returns  (pageid, summary, body_text)

    """
    return row[0], row[2], "".join(pickle.loads(row[3]))


def calculate_novelty(queue_read, queue_sql):
    """Process novelty."""

    args = queue_read.get()

    while not args is None:
        sql_args = []
        for row in args:
            pageid, summary, body_text = decode_row(row)
            (
                novelty_tokens,
                novelty_bigrams,
                novelty_trigrams,
            ) = n_gram_novelty_calculator(summary, body_text)

            sql_sub_args = {
                "pageid": pageid,
                "novelty_tokens": novelty_tokens,
                "novelty_bigrams": novelty_bigrams,
                "novelty_trigrams": novelty_trigrams,
            }

            sql_args.append(sql_sub_args)

        # Yield results
        queue_sql.put(sql_args)
        args = queue_read.get()


def main(queue_read, queue_sql):
    # Create generator to read data from SQL db
    article_generator = retrive_suitable_strings(batchsize=BATCH_SIZE_NOVELTY)

    # Create novelty calculator process
    novelty_processes = []
    for _ in range(NOVELTY_N_PROCESSES):
        p = Process(target=calculate_novelty, args=(queue_read, queue_sql))
        novelty_processes.append(p)
        p.start()

    # Create SQL processes
    sql_process = Process(
        target=insert_observations_in_table_mp,
        args=(queue_sql, WikiArticleNovelty, TEMP_DB),
    )
    sql_process.start()

    # put data into the read queue
    count_articles = 0
    # I know that there are ~1.5M suitable articles but this is not necessarilly correct
    with tqdm(total=1.5e6) as pbar:
        for args in article_generator:
            queue_read.put(args)

            # Track progress
            count_articles += BATCH_SIZE_NOVELTY
            pbar.update(BATCH_SIZE_NOVELTY)
            if count_articles % (BATCH_SIZE_NOVELTY * 1) == 0:
                print(f"{count_articles} articles processed")

    # Terminate
    for _ in range(NOVELTY_N_PROCESSES):
        queue_read.put(None)

    for p in novelty_processes:
        p.join()

    queue_sql.put(None)
    sql_process.join()


if __name__ == "__main__":
    # https://stackoverflow.com/questions/3172929/operationalerror-database-is-locked
    # Apparently SQL lite does not support concurrent operations so we cant
    # read using a generator and import to the same database since two connections
    # will be required to the database. Instead we will create a temporary db
    # and then copy the content and delete it.
    start_time = time.time()

    # Make novelty db
    queue_read = Queue(maxsize=NOVELTY_READ_QUE_SIZE)
    queue_sql = Queue(maxsize=NOVELTY_SQL_QUE_SIZE)
    main(queue_read, queue_sql)

    # Transfer to main db
    src_query = """
        SELECT *
        FROM wiki_article_novelty
    """
    transfer_to_new_db(
        src_query,
        src_db=TEMP_DB,
        dest_db=SQL_WIKI_DUMP,
        dest_table=WikiArticleNovelty,
        batch_formater=novelty_data_input_formater,
    )

    # Delete temporary db
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    end_time = time.time()
    print("finished in: ", str(timedelta(seconds=end_time - start_time)))

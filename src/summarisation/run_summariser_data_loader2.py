#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:31:17 2021

@author: ivanr
"""
import os

os.chdir("/home/ivanr/git/document_information_extraction")
os.getcwd()

os.environ["MKL_THREADING_LAYER"] = "GNU"


import torch

torch.set_num_threads(1)
# torch.multiprocessing.set_start_method("spawn")

from transformers import (
    # BertTokenizerFast as BertTokenizer,
    # BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.summarisation.data_loader2 import (
    SummariserTrainer,
    article_ids_train,
    # model,
    SummarisationDataModule,
    article_ids_test,
    # tokenizer,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


RANDOM_SEED = 42

if __name__ == "__main__":

    pl.seed_everything(RANDOM_SEED)

    BART_MODEL_NAME = "facebook/bart-large-cnn"

    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", TORCH_DEVICE)
    language = "english"

    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL_NAME, return_dict=True).to(
        TORCH_DEVICE
    )

    N_EPOCHS = 10
    BATCH_SIZE = 1

    data_module = SummarisationDataModule(
        article_ids_train,
        article_ids_test,
        tokenizer,
        batch_size=BATCH_SIZE,
    )

    # help(data_module.train_dataloader())
    # data_module.train_dataloader().__getitem__()
    # data_module

    steps_per_epoch = len(article_ids_train) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS

    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    model_pl = SummariserTrainer(model, warmup_steps, total_training_steps)
    # self = model_pl

    # =============================================================================
    # call backs
    # =============================================================================

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    logger = TensorBoardLogger("lightning_logs", name="summarisation")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    trainer = pl.Trainer(
        logger=logger,
        # checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=1,
    )

    print("fit process started")
    trainer.fit(model_pl, data_module)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:31:17 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
from pathlib import Path

WORKING_DIRECTORY = Path(os.getcwd())
sys.path.append(str(WORKING_DIRECTORY))


import torch

torch.set_num_threads(os.cpu_count())
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

from tqdm import tqdm

from transformers import (
    # get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


from src.summarisation.summarisation_trainer import (
    SummariserTrainer,
    SummarisationDataModule,
)
from src.summarisation.query_suitable_articles import get_article_ids

# =============================================================================
# Statics
# =============================================================================
from src.summarisation.summariser_statics import (
    RANDOM_SEED,
    MODEL_NAME,
    N_SAMPLE_TEXTS,
    ENCODER_MAX_LENGTH,
    # DECODER_MAX_LENGTH,
    TRAIN_TEST_SPLIT,
    N_EPOCHS,
    BATCH_SIZE,
)

if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
else:
    TORCH_DEVICE = "cpu"
    print("Device: ", TORCH_DEVICE)

# =============================================================================
# Main module
# =============================================================================


if __name__ == "__main__":

    def main():
        # --- Set random seed ---
        print("Seeding...")
        pl.seed_everything(RANDOM_SEED)

        # --- Get train and test articles ---
        print("Retrieving ids...")
        all_ids = get_article_ids(
            random_state=RANDOM_SEED,
            n_sample_texts=N_SAMPLE_TEXTS,
            max_tokens_body=ENCODER_MAX_LENGTH - 2,
        )

        article_ids_train = all_ids[: int(len(all_ids) * TRAIN_TEST_SPLIT)]
        article_ids_test = all_ids[int(len(all_ids) * TRAIN_TEST_SPLIT) :]

        steps_per_epoch = len(article_ids_train) // BATCH_SIZE
        total_training_steps = steps_per_epoch * N_EPOCHS

        warmup_steps = total_training_steps * 0  # // 5
        warmup_steps, total_training_steps

        # --- Instantiate tokenizer, model, data generator ---
        print("Instantiating objects...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, return_dict=True).to(
            TORCH_DEVICE
        )

        data_module = SummarisationDataModule(
            article_ids_train,
            article_ids_test,
            tokenizer,
            batch_size=BATCH_SIZE,
        )

        model_pl = SummariserTrainer(model, warmup_steps, total_training_steps)

        # --- Define callbacks ---
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

        # class LitProgressBar(ProgressBar):
        #     def init_train_tqdm(self):
        #         bar = super().init_validation_tqdm()
        #         bar.set_description("running train ...")
        #         return bar

        #     def init_sanity_tqdm(self):
        #         bar = super().init_validation_tqdm()
        #         bar.set_description("running sanity ...")
        #         return bar

        #     def init_test_tqdm(self):
        #         bar = super().init_validation_tqdm()
        #         bar.set_description("running test ...")
        #         return bar

        #     def init_validation_tqdm(self):
        #         bar = super().init_validation_tqdm()
        #         bar.set_description("running validation ...")
        #         return bar

        class LitProgressBar(ProgressBar):
            def init_train_tqdm(self) -> tqdm:
                """ Override this to customize the tqdm bar for training. """
                bar = tqdm(
                    desc="Training",
                    initial=self.train_batch_idx,
                    position=(2 * self.process_position),
                    disable=self.is_disabled,
                    leave=True,
                    # dynamic_ncols=False,  # This two lines are only for pycharm
                    ncols=100,
                    file=sys.stdout,
                    smoothing=0,
                )
                return bar

            def init_validation_tqdm(self) -> tqdm:
                """ Override this to customize the tqdm bar for validation. """
                # The main progress bar doesn't exist in `trainer.validate()`
                has_main_bar = self.main_progress_bar is not None
                bar = tqdm(
                    desc="Validating",
                    position=(2 * self.process_position + has_main_bar),
                    disable=self.is_disabled,
                    leave=False,
                    # dynamic_ncols=False,
                    ncols=100,
                    file=sys.stdout,
                )
                return bar

            def init_test_tqdm(self) -> tqdm:
                """ Override this to customize the tqdm bar for testing. """
                bar = tqdm(
                    desc="Testing",
                    position=(2 * self.process_position),
                    disable=self.is_disabled,
                    leave=True,
                    # dynamic_ncols=False,
                    ncols=100,
                    file=sys.stdout,
                )
                return bar

        bar = LitProgressBar()  # refresh_rate=0

        # -- Instantiate trainer ---
        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback, checkpoint_callback, bar],
            max_epochs=N_EPOCHS,
            gpus=1,
            #  progress_bar_refresh_rate=0,
            # show_progress_bar=True,
        )

        # --- Run ---
        print("Running...")
        trainer.fit(model_pl, data_module)

    main()

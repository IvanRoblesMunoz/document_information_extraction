#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 19:46:34 2021

@author: ivanr
"""

# =============================================================================
# Imports
# =============================================================================
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.summarisation.query_suitable_articles import get_article_ids
from src.data.wikipedia.wiki_data_base import retrive_observations_from_ids

# =============================================================================
# Statics
# =============================================================================
from src.summarisation.summariser_statics import (
    RANDOM_SEED,
    ENCODER_MAX_LENGTH,
    DECODER_MAX_LENGTH,
)

# =============================================================================
# Helper funtions
# =============================================================================


def decode_row(article):
    """Decodes row from SQL query to produce summary and body."""
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
    """Encode text using model tokenizer."""
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


# =============================================================================
# Generator classes
# =============================================================================
class SummaryDataset(Dataset):
    def __init__(
        self,
        article_ids: list,
        tokenizer: AutoTokenizer,
        max_input_len: int = ENCODER_MAX_LENGTH,
        max_output_len: int = DECODER_MAX_LENGTH,
    ):
        self.article_ids = article_ids
        self.tokenizer = tokenizer
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

        to_return = dict(
            input_ids=body_encoding["input_ids"].flatten(),
            attention_mask=body_encoding["attention_mask"].flatten(),
            target_ids=summary_encoding["input_ids"].flatten(),
            target_mask=summary_encoding["attention_mask"].flatten(),
        )

        return to_return


class SummarisationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ids,
        test_ids,
        tokenizer,
        batch_size=10,
        max_input_len: int = ENCODER_MAX_LENGTH,
        max_output_len: int = DECODER_MAX_LENGTH,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def setup(self, stage=None):
        self.train_dataset = SummaryDataset(
            self.test_ids, self.tokenizer, self.max_input_len, self.max_output_len
        )

        self.test_dataset = SummaryDataset(
            self.test_ids, self.tokenizer, self.max_input_len, self.max_output_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)


class SummariserTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        n_training_steps=None,
        n_warmup_steps=0,
        criterion=nn.NLLLoss(),
    ):
        super().__init__()
        self.model = model
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = criterion

    def forward(self, input_ids, attention_mask, target_ids, target_mask):

        output_dict = self.model(input_ids, attention_mask=attention_mask)
        output = output_dict["logits"]
        loss = 0

        for pred_obs, target_obs in zip(output, target_ids):
            m = nn.LogSoftmax(dim=1)
            loss += self.criterion(m(pred_obs), target_obs)

        return loss, output_dict  # output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        # with torch.no_grad():
        loss, outputs = self(input_ids, attention_mask, target_ids, target_mask)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        with torch.no_grad():
            loss, outputs = self(input_ids, attention_mask, target_ids, target_mask)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        with torch.no_grad():
            loss, outputs = self(input_ids, attention_mask, target_ids, target_mask)

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def training_epoch_end(self, outputs):
        """Function to run at the end of an epoch."""
        print("Epoch ended...")

        pass
        # labels = []
        # predictions = []
        # for output in outputs:
        #     for out_labels in output["labels"].detach().cpu():
        #         labels.append(out_labels)
        #     for out_predictions in output["predictions"].detach().cpu():
        #         predictions.append(out_predictions)

        # labels = torch.stack(labels).int()
        # predictions = torch.stack(predictions)

        # for i, name in enumerate(LABEL_COLUMNS):
        #     class_roc_auc = auroc(predictions[:, i], labels[:, i])
        #     self.logger.experiment.add_scalar(
        #         f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
        #     )

    def configure_optimizers(self):
        """Configure optimizer and its parameters."""

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )


# TODO: Figure out how to decode output
# TODO: Fill in on epoch end function
# TODO: Add rouge metric
# TODO: Control summary output length and add length penalty
# TODO: Figure out why only validation loss logging is showing
# TODO: Add summariser to data loader
# TODO: Freeze layers
# TODO: learn how to reload model from checkpoint and use it for prediction

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


from src.summarisation.data_query_summariser import get_article_ids
from src.data.wikipedia.wiki_data_base import retrive_observations_from_ids

# =============================================================================
# Statics
# =============================================================================
RANDOM_SEED = 42

ENCODER_MAX_LENGTH = 1024
DECODER_MAX_LENGTH = 1024


print("Reading data")
# =============================================================================
# Data
# =============================================================================
all_ids = get_article_ids(
    random_state=RANDOM_SEED, n_sample_texts=10000, max_tokens_body=1022
)

article_ids_train = all_ids[: int(len(all_ids) * 0.9)]
article_ids_test = all_ids[int(len(all_ids) * 0.9) :]


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


# train_data = convert_ids_to_features([article_ids_train[1]])
# sample_summary = encode(tokenizer, train_data["summary"][0], DECODER_MAX_LENGTH)

# print(sample_summary.keys())
# print(sample_summary["input_ids"].shape)
# print(sample_summary["attention_mask"].shape)

# print(tokenizer.convert_ids_to_tokens(sample_summary["input_ids"].squeeze())[:20])

import pickle


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
        with open("output_ref.pkl", "wb") as f:
            pickle.dump(to_return, f)

        return to_return


# with open("output.pkl", "rb") as f:
#     to_return = pickle.load(f)

# dataset = SummaryDataset(article_ids=article_ids_train, tokenizer=tokenizer)

# sample_item = dataset[1]
# sample_item.keys()

# sample_item["input_ids"].shape


# output = model(sample_item["input_ids"], sample_item["attention_mask"])
# type(output)

# # TODO: Figure out how to decode this
# for i in output.keys():
#     try:
#         print(i, output[i].shape)
#     except AttributeError:
#         print(i, len(output[i]))

# len(output)

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# writer.add_graph(model, sample_item["input_ids"], sample_item["attention_mask"])


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


# N_EPOCHS = 10
# BATCH_SIZE = 15

# data_module = SummarisationDataModule(
#     article_ids_train,
#     article_ids_test,
#     tokenizer,
#     batch_size=BATCH_SIZE,
# )

# input_ = torch.randn(3, 5, requires_grad=True)
# input_.shape

# target_ = torch.tensor([1, 0, 4])
# target_.shape

# loss = nn.NLLLoss()
# m = nn.LogSoftmax(dim=1)
# input_ = m(input_)

# output = loss(m(input_), target_)
# output.backward()

# TODO: get criterion correctly
# losses = loss = nn.NLLLoss()

# import pickle


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
        # to_ret = [input_ids, attention_mask, target_ids, target_mask]
        # if input_ids.shape == torch.Size([2, 1, 1024]):
        #     with open("output.pkl", "wb") as f:
        #         pickle.dump(to_ret, f)

        # input_ids.shape == torch.Size([2, 1, 1024])
        # with open("output.pkl", "rb") as f:
        #     input_ids, attention_mask, target_ids, target_mask = pickle.load(f)

        # with open("output_ref.pkl", "rb") as f:
        #     to_return = pickle.load(f)
        # input_ids, attention_mask, target_ids, target_mask = to_return.values()

        # input_ids = input_ids.to(device="cuda")
        # attention_mask = attention_mask.to(device="cuda")
        # target_ids = target_ids.to(device="cuda")
        # target_mask = target_mask.to(device="cuda")
        # self.model.to("cuda")
        # print("Started forward step...")
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(target_ids.shape)
        # print(target_mask.shape)

        output_dict = self.model(input_ids, attention_mask=attention_mask)
        print("model prediction made...")
        # print(output_dict)
        output = output_dict["logits"]
        print("Forward step loss calculation...")
        loss = 0

        target_ids.shape
        output.shape
        for pred_obs, target_obs in zip(output, target_ids):
            m = nn.LogSoftmax(dim=1)
            loss += self.criterion(m(pred_obs), target_obs)

            # target_obs.shape
            # target_obs[0].shape
            # pred_obs.shape
            # torch.sum(torch.exp(m(pred_obs)))

        print("Forward step returning...")
        return loss, output_dict  # output

    def training_step(self, batch, batch_idx):
        print("Starting training step ...")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        # with torch.no_grad():
        loss, outputs = self(input_ids, attention_mask, target_ids, target_mask)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        print("Ending training step ...")
        return {
            "loss": loss,
            # "predictions": outputs,
            # "target_ids": target_ids,
            # "target_mask": target_mask,
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        with torch.no_grad():
            loss, outputs = self(input_ids, attention_mask, target_ids, target_mask)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        with torch.no_grad():
            loss, outputs = self(input_ids, attention_mask, target_ids, target_mask)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
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

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )


print("Ran all module")
# sample_item.keys()
# batch = sample_item
# input_ids = sample_item["input_ids"]
# attention_mask = sample_item["attention_mask"]
# target_ids = sample_item["target_ids"]
# target_mask = sample_item["target_mask"]


# steps_per_epoch = len(article_ids_train) // BATCH_SIZE
# total_training_steps = steps_per_epoch * N_EPOCHS

# warmup_steps = total_training_steps // 5
# warmup_steps, total_training_steps

# model_pl = SummariserTrainer(model, warmup_steps, total_training_steps)
# # =============================================================================
# # call backs
# # =============================================================================
# checkpoint_callback = ModelCheckpoint(
#     dirpath="checkpoints",
#     filename="best-checkpoint",
#     save_top_k=1,
#     verbose=True,
#     monitor="val_loss",
#     mode="min",
# )

# logger = TensorBoardLogger("lightning_logs", name="summarisation")
# early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)


# trainer = pl.Trainer(
#     logger=logger,
#     # checkpoint_callback=checkpoint_callback,
#     callbacks=[early_stopping_callback, checkpoint_callback],
#     max_epochs=N_EPOCHS,
#     gpus=1,
#     progress_bar_refresh_rate=30,
# )


# trainer.fit(model_pl, data_module)

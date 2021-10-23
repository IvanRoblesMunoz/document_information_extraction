#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 23:27:35 2021

@author: ivanr
"""

# =============================================================================
# Statics to retrieve suitable articles
# =============================================================================
from src.data.data_statics import (
    MIN_SEMANTIC_SIMILARITY,
    MIN_NOVELTY,
    MAX_NOVELTY,
    MAX_TOKENS_BODY,
)


N_SAMPLE_TEXTS = 100

# =============================================================================
# Summariser model statics
# =============================================================================
RANDOM_SEED = 0
TRAIN_TEST_SPLIT = 0.8
ENCODER_MAX_LENGTH = 1024
DECODER_MAX_LENGTH = 1024

MODEL_NAME = "facebook/bart-large-cnn"
N_EPOCHS = 10
BATCH_SIZE = 2

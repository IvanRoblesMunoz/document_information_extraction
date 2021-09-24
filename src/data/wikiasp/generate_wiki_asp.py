#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 00:30:08 2021

@author: ivanr
"""

from src.data.data_statics import RAW_DATA_PATH
import json



with open(RAW_DATA_PATH / "Animal/valid.jsonl", "r") as json_file:
    json_list = list(json_file)
    
    
all_entries = dict()
for idx, json_str in enumerate(json_list[:50]):
    result = json.loads(json_str)
    all_entries[idx] = result
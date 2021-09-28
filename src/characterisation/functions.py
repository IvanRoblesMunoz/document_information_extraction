#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:42:22 2021

@author: ivanr
"""
# pylint: disable=E1101
# =============================================================================
# Imports
# =============================================================================
import itertools
from collections import defaultdict
import pickle
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from gensim.summarization.summarizer import summarize


from src.data.data_statics import (
    BERTSCORE_MODEL,
    BERTSCORE_MODEL_LAYER,
    BERTSCORE_ALL_LAYERS,
    BERTSCORE_LANGUAGE,
    MAX_MODEL_SQUENCE_LENGTH,
)


# =============================================================================
# Statics
# =============================================================================
from src.data.data_statics import (
    MODEL_TYPE_SEMANTIC_SIMILARITY,
    BATCH_SIZE_SEMANTIC_SIMILARITY,
)

ENGLISH_STOPWORDS = set(stopwords.words("english"))
# nltk.download("wordnet")
LEMMATIZER = nltk.wordnet.WordNetLemmatizer()
SEMANTIC_SIMILARITY_MODEL = SentenceTransformer(MODEL_TYPE_SEMANTIC_SIMILARITY)

# =============================================================================
# Functions
# =============================================================================


def n_gram_novelty_calculator(summary: str, body: str) -> tuple:
    """
    Calculate novelty for n-grams 1 to 3.

    Calculates summary novelty by comparing the percentage of new tokens
    in the summary that dont appear in the body.
    This is done using a lemmatizer so that grammatical inflections do not
    affect the text.
    Furthermore, stopwords are removed.

    Parameters
    ----------
    summary : str
    body : str

    Returns
    -------
    float
        As fractions  (novelty_tokens, novelty_bigrams, novelty_trigrams).

    """
    summary = nltk.word_tokenize(summary)
    body = nltk.word_tokenize(body)

    summary = [
        LEMMATIZER.lemmatize(word) for word in summary if word not in ENGLISH_STOPWORDS
    ]
    body = [
        LEMMATIZER.lemmatize(word) for word in body if word not in ENGLISH_STOPWORDS
    ]

    # Generate sets of bigrams, trigrams and tokens for body and summary
    tokens_summary = set(summary)
    tokens_body = set(body)
    bigrams_summary = set(ngrams(summary, 2))
    bigrams_body = set(ngrams(body, 2))
    trigrams_summary = set(ngrams(summary, 3))
    trigrams_body = set(ngrams(body, 3))

    # Calculate novelty
    novel_tokens = tokens_summary - tokens_body
    novelty_tokens = len(novel_tokens) / len(tokens_summary)
    novel_bigrams = bigrams_summary - bigrams_body
    novelty_bigrams = len(novel_bigrams) / len(bigrams_summary)
    novel_trigrams = trigrams_summary - trigrams_body
    novelty_trigrams = len(novel_trigrams) / len(trigrams_summary)

    return novelty_tokens, novelty_bigrams, novelty_trigrams


# =============================================================================
# Semantic
# =============================================================================
from src.data.wikipedia.wiki_data_base import retrive_suitable_strings

batch_generator = retrive_suitable_strings(limit=100, batchsize=5)

for batch in batch_generator:
    pass


def prepare_batch_data(batch):
    """
    Prepare data batch.

    Perform extractive summarisation of body to fit into our model without
    truncating it.
    Also pass output as required by the function.

    """
    # Retrieve (id, summary, text)
    to_encode = []
    id_list = []
    for row in batch:
        id_list.append(row[0])
        summary = row[2]
        body = "".join(pickle.loads(row[3]))
        try:
            # If there is only one sentence a ValueError will be thrown
            body = summarize(body, word_count=MAX_MODEL_SQUENCE_LENGTH)
        except ValueError:
            pass
        to_encode += [summary]
        to_encode += [body]

    encoding_batch = to_encode, id_list
    return encoding_batch


def compute_semantic_similarity(encoding_batch):
    """
    Computes semantic similarity between summary and body.

    We cant use this for documents were inputs are larger than 512 tokens.
    Since they will be truncated. Thus, we will use text rank instead to
    produce an extractive summary of suitable length.

    """

    to_encode, id_list = encoding_batch

    # Embbed all texts
    all_embeddings = SEMANTIC_SIMILARITY_MODEL.encode(to_encode, batch_size=100)
    query_embeddings, passage_embeddings = all_embeddings[::2], all_embeddings[1::2]

    all_similarity = []
    for idx, (q_emb, p_emb) in enumerate(zip(query_embeddings, passage_embeddings)):
        semantic_similarity = util.dot_score(q_emb, p_emb).numpy()[0][0]
        pageid = id_list[idx]
        all_similarity.append(
            {"pageid": pageid, "semantic_similarity": semantic_similarity}
        )
    return all_similarity


# =============================================================================
# Bertscore
# =============================================================================
# import time
# from datetime import timedelta
# from itertools import zip_longest
# import torch
# from bert_score.utils import (
#     get_bert_embedding,
#     pad_sequence,
#     greedy_cos_idf,
#     lang2model,
#     model2layers,
#     get_tokenizer,
#     get_model,
# )

# def pad_batch_stats(sen_batch, stats_dict, device):
#     stats = [stats_dict[s] for s in sen_batch]
#     emb, idf = zip(*stats)
#     emb = [e.to(device) for e in emb]
#     idf = [i.to(device) for i in idf]
#     lens = [e.size(0) for e in emb]
#     emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
#     idf_pad = pad_sequence(idf, batch_first=True)

#     def length_to_mask(lens):
#         lens = torch.tensor(lens, dtype=torch.long)
#         max_len = max(lens)
#         base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
#         return base < lens.unsqueeze(1)

#     pad_mask = length_to_mask(lens).to(device)
#     return emb_pad, pad_mask, idf_pad


# class BERTScore:
#     """
#     The default model is as shown bellow with the following performances.

#     model: allenai/led-base-16384
#     best layer:6
#     corr:0.7122
#     rank:29
#     max tokens:16382

#     It would take arounds 62 hours to compute bertscore for our suitable
#     ~1.4M articles, so we will not use it.

#     """

#     def __init__(
#         self,
#         lang: str = BERTSCORE_LANGUAGE,
#         model_type: str = BERTSCORE_MODEL,
#         num_layers: int = BERTSCORE_MODEL_LAYER,
#         all_layers: bool = BERTSCORE_ALL_LAYERS,
#         rescale_with_baseline: bool = False,
#         device: str = None,
#         verbose: bool = True,
#     ):
#         self.lang = lang
#         self.model_type = model_type
#         self.num_layers = num_layers
#         self.all_layers = all_layers
#         self.rescale_with_baseline = rescale_with_baseline
#         self.device = device
#         self.verbose = verbose

#         self._perform_attribute_checks()
#         self._load_model_related_attributes()

#     def _perform_attribute_checks(self):
#         if self.rescale_with_baseline:
#             assert (
#                 self.lang is not None
#             ), "Need to specify Language when rescaling with baseline"

#     def _load_model_related_attributes(self):

#         if self.model_type is None:
#             self.lang = self.lang.lower()
#             self.model_type = lang2model[self.lang]

#         if self.num_layers is None:
#             self.num_layers = model2layers[self.model_type]

#         # Get model and tokenizer
#         self.tokenizer = get_tokenizer(self.model_type)
#         self.model = get_model(self.model_type, self.num_layers, self.all_layers)

#         if self.device is None:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             self.model.to(self.device)

#         self._idf_dict = defaultdict(lambda: 1.0)
#         self._idf_dict[self.tokenizer.sep_token_id] = 0
#         self._idf_dict[self.tokenizer.cls_token_id] = 0

#     def _produce_embedings(self, sentences, batch_name, batch_size=1):
#         embs = []
#         iter_range = range(0, len(sentences), batch_size)
#         if self.verbose:
#             print(f"computing bert embedding for {batch_name}.")
#             iter_range = iter_range
#         stats_dict = dict()

#         for batch_start in iter_range:

#             sen_batch = sentences[batch_start : batch_start + batch_size]
#             embs, masks, padded_idf = get_bert_embedding(
#                 sen_batch,
#                 self.model,
#                 self.tokenizer,
#                 self._idf_dict,
#                 device=self.device,
#                 all_layers=self.all_layers,
#             )

#             embs = embs.cpu()
#             masks = masks.cpu()
#             padded_idf = padded_idf.cpu()
#             for i, sen in enumerate(sen_batch):
#                 sequence_len = masks[i].sum().item()
#                 emb = embs[i, :sequence_len]
#                 idf = padded_idf[i, :sequence_len]
#                 stats_dict[sen] = (emb, idf)

#         return stats_dict

#     def make_embedding_dictionary(
#         self, refs: list, hyps: list, batch_size_ref: int = 20, batch_size_hyp: int = 20
#     ) -> dict:
#         """
#         Make a dictionary containing sentence embeddings.
#         Parameters
#         ----------
#         refs : list[list[str]]
#             Reference sentences.
#         hyps : list[list[str]]
#             Candidate sentences.
#         batch_size_ref : int, optional
#             Batch size to use during reference embedding. The default is 20.
#         batch_size_hyp : int, optional
#             Batch size to use during candidate embedding.. The default is 20.
#         Returns
#         -------
#         dict
#             Dictionary containing sentence as keys and embeddings as values.
#         """
#         refs_flat = list(itertools.chain(*refs))
#         hyps_flat = list(itertools.chain(*hyps))
#         with torch.no_grad():
#             stats_dict = self._produce_embedings(
#                 refs_flat, batch_size=batch_size_ref, batch_name="reference sentences"
#             )
#             stats_dict.update(
#                 self._produce_embedings(
#                     hyps_flat,
#                     batch_size=batch_size_hyp,
#                     batch_name="candidate sentences",
#                 )
#             )
#         return stats_dict

#     def bert_cos_score_idf(
#         self, refs: str, hyps: str, stats_dict: dict, verbose: bool = True
#     ):
#         """
#         Produce final bert score.
#         Parameters
#         ----------
#         refs : str
#             Reference sentence.
#         hyps : str
#             Candidate sentence.
#         stats_dic : dict
#             Dictionary containg sentence embedings.
#         verbose : bool, optional
#             Verbose. The default is True.
#         Returns
#         -------
#         preds : TYPE
#             array[P, R, F].
#         """
#         preds = []

#         device = next(self.model.parameters()).device

#         with torch.no_grad():
#             batch_refs = refs
#             batch_hyps = hyps
#             ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
#             hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)

#             P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, self.all_layers)
#             preds.append(torch.stack((P, R, F1), dim=-1).cpu())

#         preds = torch.cat(preds, dim=1 if self.all_layers else 0)
#         return preds.numpy()[0]


# # =============================================================================
# # Test Bertscore time
# # =============================================================================
# from src.data.wikipedia.wiki_data_base import retrive_suitable_strings

# batch_generator = retrive_suitable_strings(limit=100, batchsize=5)


# bertscorer = BERTScore()
# start_time = time.time()
# for batch in batch_generator:
#     batch_dictionary = {"pageid": [], "summary": [], "body": [], "predictions": []}
#     for row in batch:
#         batch_dictionary["pageid"].append(row[0])
#         batch_dictionary["summary"].append([row[2]])
#         batch_dictionary["body"].append(["".join(pickle.loads(row[3]))])


#     refs = batch_dictionary["summary"]
#     cands = batch_dictionary["body"]
#     article_ids = batch_dictionary["pageid"]

#     load_time = time.time()
#     stats_dict = bertscorer.make_embedding_dictionary(refs, cands)
#     embedding_time = time.time()

#     for idx, sumry, txt in zip(article_ids, refs, cands):
#         preds = bertscorer.bert_cos_score_idf(sumry, txt, stats_dict)
#         batch_dictionary["predictions"].append(preds)

#     prediction_time = time.time()

#     print(
#         f"""
#         load time = {load_time-start_time}
#         embedding_time = {embedding_time-load_time}
#         prediction_time = {prediction_time-embedding_time}
#     """
#     )
#     del stats_dict
#     torch.cuda.empty_cache()

# endtime = time.time()
# print(f"Total time: {endtime-start_time}")

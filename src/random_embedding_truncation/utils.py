import os
import tomllib
from typing import Any

import numpy as np


def convert_dataset_name_for_nanobeir(dataset_name: str) -> str:
    if dataset_name == "climate-fever":
        dataset_name = "climatefever"
    if dataset_name == "dbpedia-entity":
        dataset_name = "dbpedia"
    if dataset_name == "fiqa":
        dataset_name = "fiqa2018"
    if dataset_name == "quora":
        dataset_name = "quoraretrieval"
    if dataset_name == "webis-touche2020":
        dataset_name = "touche2020"
    return dataset_name


def read_toml(toml_file: str) -> dict[str, Any]:
    if not os.path.isfile(toml_file):
        raise FileNotFoundError(f"Not Found: {toml_file}")
    with open(toml_file, "rb") as f:
        return tomllib.load(f)


def make_mask(scale: float, original_dimension: int) -> list[int]:
    return np.random.choice(
        original_dimension, int(original_dimension * scale), replace=False
    ).tolist()


def get_nanobeir_task_ids() -> list[str]:
    return [
        "NanoBEIR",
        "NanoArguAna",
        "NanoClimateFEVER",
        "NanoDBPedia",
        "NanoFEVER",
        "NanoFiQA2018",
        "NanoHotpotQA",
        "NanoMSMARCO",
        "NanoNFCorpus",
        "NanoNQ",
        "NanoQuoraRetrieval",
        "NanoSCIDOCS",
        "NanoSciFact",
        "NanoTouche2020",
    ]


def get_mteb_task_ids() -> list[str]:
    return [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ]


def get_beir_task_ids() -> list[str]:
    return [
        "arguana",
        "climate-fever",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        "msmarco",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
    ]

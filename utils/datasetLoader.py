from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path


def load_wildguard():
    wildguard_test = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    wildguard_test = wildguard_test.to_pandas().dropna()
    prompts = wildguard_test["prompt"].values.tolist()
    labels = [{"unharmful": 0, "harmful": 1}[x] for x in wildguard_test["prompt_harm_label"].values.tolist()]
    return prompts, labels, wildguard_test


def loadToxicChat():
    toxic_chat = load_dataset("lmsys/toxic-chat", "toxicchat0124")["test"].to_pandas()
    prompts = toxic_chat["user_input"].values.tolist()
    labels = toxic_chat["toxicity"].values.tolist()
    return prompts, labels


def loadAegis():
    def majority_vote(row):
        votes = [row[f"labels_{i}"] for i in range(5)]
        return pd.Series(votes).mode()[0]

    aegis_df = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")["test"].to_pandas()
    aegis_df = aegis_df[aegis_df["text_type"] == "user_message"].copy()
    aegis_df["voting"] = aegis_df.apply(majority_vote, axis=1)

    texts = aegis_df["text"].values.tolist()
    labels = [0 if x == "Safe" else 1 for x in aegis_df["voting"].values.tolist()]
    return texts, labels


def loadOrBenchHard():
    or_bench_df = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"].to_pandas()
    prompts = or_bench_df["prompt"].values.tolist()
    labels = [0] * len(prompts)
    return prompts, labels


def loadRemedyTest(fold: int | str = 0):
    """Load the REMEDy test split for a given cross-validation fold."""
    data_path = Path(__file__).resolve().parent.parent / "dataset" / f"fold-{fold}" / "test.json"
    test = json.load(open(data_path))
    prompts, labels = [], []
    for item in test:
        prompts.append(item["text"])
        labels.append(1 if item["macro_label"] == "MALICIOUS" else 0)
    return prompts, labels


# Convenience dict used by inference scripts
DATASETS_LOADER = {
    "Remedy": loadRemedyTest,  # call with fold= kwarg
    "WildGuard": load_wildguard,
    "ToxicChat": loadToxicChat,
    "Aegis": loadAegis,
    "OrBench": loadOrBenchHard,
}

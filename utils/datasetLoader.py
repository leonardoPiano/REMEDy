from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path
def load_wildguard():
    wildguard_test = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    wildguard_test = wildguard_test.to_pandas()
    wildguard_test = wildguard_test.dropna()

    prompts = wildguard_test["prompt"].values.tolist()
    labels = wildguard_test["prompt_harm_label"].values.tolist()
    map2id = {"unharmful": 0, "harmful": 1}
    labels= [map2id[x] for x in labels]
    return prompts,labels,wildguard_test

def loadToxicChat():
    toxic_chat = load_dataset("lmsys/toxic-chat", "toxicchat0124")["test"]
    toxic_chat_df = toxic_chat.to_pandas()
    prompts = toxic_chat_df["user_input"].values.tolist()
    labels= toxic_chat_df["toxicity"].values.tolist()
    return prompts,labels

def loadAegis():
    def majority_vote(row):
        # Prende i valori dalle colonne di interesse
        labels = [row['labels_0'], row['labels_1'], row['labels_2'], row['labels_3'], row['labels_4']]
        # Calcola la moda (valore più frequente)
        return pd.Series(labels).mode()[0]

    aegis = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")
    aegis_df = aegis["test"].to_pandas()
    aegis_df = aegis_df[aegis_df["text_type"] == "user_message"]
    aegis_df["voting"] = aegis_df.apply(majority_vote, axis=1)

    text = aegis_df["text"].values.tolist()
    labels = aegis_df["voting"].values.tolist()
    binary_labels = []
    for x in labels:
        if x == "Safe":
            binary_labels.append(0)
        else:
            binary_labels.append(1)
    return  text,binary_labels

def loadOrBenchHard():
    or_bench = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")
    or_bench_df=or_bench["train"].to_pandas()
    prompts=or_bench_df["prompt"].values.tolist()
    labels=[0]*len(prompts)
    return prompts,labels

def loadRemedyTest():
    current_file_path = Path(__file__).resolve()
    data_path = current_file_path.parent.parent / "dataset" /"gold"/ "test.json"
    test=json.load(open(data_path))
    prompts,labels=[],[]
    for item in test:
        prompts.append(item["text"])
        if item["macro_label"]=="MALICIOUS":
           labels.append(1)
           
        else:        
          labels.append(0)
            
    return prompts,labels

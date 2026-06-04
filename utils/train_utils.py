import json
import os
from pathlib import Path

# Load schema relative to this file 
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "REMEdy_schema.json"
schema = json.load(open(_SCHEMA_PATH))

malicious = list(schema["MALICIOUS"]["categories"].keys())
targets = list(schema["TARGETS"]["categories"].keys())
neutrals = list(schema["NEUTRAL"]["categories"].keys())

INSTRUCTION = """\
You are a moderation model.
Classify the text as [MALICIOUS, BENIGN, DOUBTFUL] and extract rationale spans according to the taxonomy.
Taxonomy:
MALICIOUS = {malicious}
TARGETS = {target}

---
### Output format:
"global_label": "MALICIOUS | BENIGN | DOUBTFUL",
"rationales": span (label);; ..
---
Now annotate:

""".format(malicious=malicious, target=targets)

# Model shortname → HuggingFace id
MODEL_DICT = {
    "llama3.1-8": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-3": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "gemma2": "unsloth/gemma-2-9b-it",
}

# unsloth variant ids (used during inference with vLLM)
MODEL_DICT_UNSLOTH = {
    "llama3.1-8": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-3": "unsloth/Llama-3.2-3B-Instruct",
    "mistral": "unsloth/mistral-7b-v0.3",
    "gemma2": "unsloth/gemma-2-9b-it",
}





def create_conversation_dataset(dataset: list) -> dict:
    """Convert a list of annotated examples into the sharegpt conversation format."""
    conv_data = []
    for item in dataset:
        text = item["text"]
        annotations = item["rationales"]
        label = item["macro_label"]
        prompt = INSTRUCTION + " " + text
        if len(annotations) > 0:
            answer = ";; ".join([f"{x['span']} ({x['label']})" for x in annotations])
        else:
            answer = "[]"
        response = f"global_label: {label}\nrationales: {answer}"
        conv = [{"from": "human", "value": prompt}, {"from": "gpt", "value": response}]
        conv_data.append(conv)
    return {"conversations": conv_data}

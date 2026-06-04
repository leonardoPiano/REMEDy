"""ShieldGemma baseline inference."""
import os
import json
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.datasetLoader import load_wildguard, loadToxicChat, loadAegis, loadOrBenchHard, loadRemedyTest
from .prompts.shieldgemma_guide import guideline


MODEL_ID = "google/shieldgemma-9b"
MODEL_NAME = "shieldgemma"


def run(fold: int = 0, cuda_device: str = "0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    out_path = f"output/fold-{fold}/parsed/GUARD/{MODEL_NAME}"
    os.makedirs(out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    vocab = tokenizer.get_vocab()

    def predict(prompt: str) -> int:
        chat = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            chat, guideline=guideline, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        selected = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        prob_yes = torch.softmax(selected, dim=0)[0].item()
        return int(prob_yes > 0.5)

    datasets = {
        "Remedy": loadRemedyTest(fold),
        "WildGuard": load_wildguard(),
        "ToxicChat": loadToxicChat(),
        "Aegis": loadAegis(),
        "OrBench": loadOrBenchHard(),
    }

    for name, values in datasets.items():
        print(f"Processing {name}")
        prompts, labels, *_ = values
        predictions = [
            {"text": p, "pred": predict(p), "real": l}
            for p, l in tqdm(zip(prompts, labels), total=len(prompts))
        ]
        out_file = os.path.join(out_path, f"{name}.json")
        with open(out_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"  → saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()
    run(fold=args.fold, cuda_device=args.cuda)

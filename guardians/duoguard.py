"""DuoGuard baseline inference (all three model variants)."""
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.datasetLoader import load_wildguard, loadToxicChat, loadAegis, loadOrBenchHard, loadRemedyTest


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "duo_0.5":    ("DuoGuard/DuoGuard-0.5B",                    "Qwen/Qwen2.5-0.5B"),
    "duo_1.5":    ("DuoGuard/DuoGuard-1.5B-transfer",            "Qwen/Qwen2.5-1.5B"),
    "duo_llama":  ("DuoGuard/DuoGuard-1B-Llama-3.2-transfer",    "meta-llama/Llama-3.2-1B"),
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(model_id: str, tokenizer_id: str, cuda_device: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, token=HF_TOKEN
    ).to(f"cuda:{cuda_device}")
    return tokenizer, model


def predict(prompt: str, tokenizer, model, device: str, threshold: float = 0.5) -> int:
    """Return 1 (unsafe) or 0 (safe) for a single prompt."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(f"cuda:{device}")
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, num_categories)
        probabilities = torch.sigmoid(logits)
    max_prob = probabilities[0].max().item()
    return 1 if max_prob > threshold else 0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(variant: str = "duo_llama", fold: int = 0, cuda_device: str = "0", threshold: float = 0.5):
    if variant not in MODEL_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(MODEL_REGISTRY)}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    model_id, tokenizer_id = MODEL_REGISTRY[variant]
    print(f"Loading {variant}  →  {model_id}")
    tokenizer, model = load_model(model_id, tokenizer_id, cuda_device)

    out_path = f"output/fold-{fold}/parsed/GUARD/{variant}"
    os.makedirs(out_path, exist_ok=True)

    datasets = {
        "Remedy":    loadRemedyTest(fold),
        "WildGuard": load_wildguard(),
        "ToxicChat": loadToxicChat(),
        "Aegis":     loadAegis(),
        "OrBench":   loadOrBenchHard(),
    }

    for name, values in datasets.items():
        out_file = os.path.join(out_path, f"{name}.json")
        if os.path.exists(out_file):
            print(f"  Skipping {name} (already exists)")
            continue

        print(f"  Processing {name}")
        prompts, labels, *_ = values
        results = [
            {"text": p, "pred": predict(p, tokenizer, model, cuda_device, threshold), "real": l}
            for p, l in tqdm(zip(prompts, labels), total=len(prompts))
        ]
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    → saved {out_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DuoGuard baseline inference.")
    parser.add_argument(
        "--variant", type=str, default="duo_llama",
        choices=list(MODEL_REGISTRY),
        help=(
            "Model variant to run. "
            "duo_0.5 = DuoGuard-0.5B, "
            "duo_1.5 = DuoGuard-1.5B-transfer, "
            "duo_llama = DuoGuard-1B-Llama-3.2-transfer."
        ),
    )
    parser.add_argument("--all", action="store_true", help="Run all three variants sequentially.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variants = list(MODEL_REGISTRY) if args.all else [args.variant]
    for v in variants:
        print(f"\n{'='*60}\nVariant: {v}\n{'='*60}")
        run(variant=v, fold=args.fold, cuda_device=args.cuda, threshold=args.threshold)
    print("\nDone.")

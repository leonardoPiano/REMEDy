"""LlamaGuard baseline inference."""
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
from vllm import LLM, SamplingParams

from utils.datasetLoader import load_wildguard, loadToxicChat, loadAegis, loadOrBenchHard, loadRemedyTest


MODEL_ID = "meta-llama/Llama-Guard-3-8B"
MODEL_NAME = "llamaguard"


def _parse_answer(text: str) -> int:
    return 1 if "unsafe" in text.lower() else 0


def run(fold: int = 0, cuda_device: str = "1"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    out_path = f"output/fold-{fold}/parsed/GUARD/{MODEL_NAME}"
    os.makedirs(out_path, exist_ok=True)

    llm = LLM(model=MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=500)

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
        messages = [[{"role": "user", "content": x}] for x in prompts]
        outputs = llm.chat(messages, sampling_params)

        predictions = [
            {"text": prompts[i], "pred": _parse_answer(out.outputs[0].text), "real": labels[i]}
            for i, out in enumerate(outputs)
        ]

        out_file = os.path.join(out_path, f"{name}.json")
        with open(out_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"  → saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="1")
    args = parser.parse_args()
    run(fold=args.fold, cuda_device=args.cuda)

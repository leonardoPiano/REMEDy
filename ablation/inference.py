"""Ablation inference script.

Variants:
  --ablation no_rationale   uses models from ablation/models/.../no_rationale/
                            saves to ablation/output/fold-{fold}/parsed/FT/{model}/
  --ablation no_harmless    uses models from ablation/models/.../no_harmless/
                            saves to ablation/output/fold-{fold}/parsed/FT/abl_harmless/{model}/

The output paths mirror what ablation/PromptClassification.ipynb expects:
  ./output/fold-0/parsed/FT/{model}/              (no_rationale)
  ./output/fold-0/parsed/FT/abl_harmless/{model}/ (no_harmless)

Run from the project root:
  python ablation/inference.py --model llama3.2-3 --fold 0 --ablation no_rationale
  python ablation/inference.py --model gemma2 --fold 0 --ablation no_harmless
"""
import os
import sys
import json
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.train_utils import INSTRUCTION, MODEL_DICT_UNSLOTH
from utils.datasetLoader import load_wildguard, loadToxicChat, loadAegis, loadOrBenchHard, loadRemedyTest
from utils.parsing import parse_answer_rationale, label_to_binary


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation inference for REMEDy.")
    parser.add_argument("--model", type=str, default="llama3.2-3", choices=list(MODEL_DICT_UNSLOTH.keys()))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["no_rationale", "no_harmless"])
    parser.add_argument("--checkpoint", type=str, default="checkpoint-640")
    return parser.parse_args()


def main():
    args = parse_args()
    fold = args.fold
    model_name = MODEL_DICT_UNSLOTH[args.model]
    adapter_path = f"ablation/models/fold-{fold}/{args.ablation}/{args.model}/{args.checkpoint}"

    if args.ablation == "no_harmless":
        output_dir = f"ablation/output/fold-{fold}/parsed/FT/abl_harmless/{args.model}"
    else:
        output_dir = f"ablation/output/fold-{fold}/parsed/FT/no_rationale/{args.model}"

    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(model=model_name, dtype=torch.bfloat16, trust_remote_code=True,
               enable_lora=True, max_lora_rank=64)

    datasets = {
        "Remedy": loadRemedyTest(fold),
        "WildGuard": load_wildguard(),
        "ToxicChat": loadToxicChat(),
        "Aegis": loadAegis(),
        "OrBench": loadOrBenchHard(),
    }

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=500,repetition_penalty=1.1)

    for name, values in datasets.items():
        out_file = os.path.join(output_dir, f"{name}.json")
        if os.path.exists(out_file):
            print(f"  Skipping {name} (already exists)")
            continue

        prompts, labels, *_ = values
        print(f"  Processing {name} ({len(prompts)} examples)")

        messages = [[{"role": "user", "content": INSTRUCTION + " " + x}] for x in prompts]
        outputs = llm.chat(messages, sampling_params,
                           lora_request=LoRARequest("adapter", 1, adapter_path))

        predictions = []
        for i, out in enumerate(outputs):
            label_str, rationales = parse_answer_rationale(out.outputs[0].text)
            predictions.append({
                "text": prompts[i],
                "pred": label_to_binary(label_str),
                "real": labels[i],
                "label": label_str,
                "rationales": rationales,
            })

        with open(out_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"    → saved {out_file}")


if __name__ == "__main__":
    main()

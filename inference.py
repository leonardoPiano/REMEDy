"""Inference script: FT and ZERO-SHOT across all datasets.

Modes:
  --mode FT    fine-tuned LoRA adapter
  --mode ZERO  zero-shot base model

Datasets:
  standard  Remedy, WildGuard, ToxicChat, Aegis, OrBench
  hatexplain HateXplain rationale extraction
  cosafe     CoSafe multi-turn conversations
  all        all of the above (default)

Usage examples:
  python inference.py --model llama3.2-3 --fold 0 --mode FT --dataset all
  python inference.py --model gemma2 --mode ZERO --dataset standard
  python inference.py --model mistral --fold 0 --mode FT --dataset hatexplain
  python inference.py --model llama3.1-8 --mode ZERO --dataset cosafe
"""
import os
import json
import argparse
import re
from glob import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dotenv import load_dotenv

load_dotenv()

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.train_utils import INSTRUCTION, MODEL_DICT_UNSLOTH
from utils.prompts import ZERO_SYSTEM_CLASSIFICATION,MULTI_TURN_INSTRUCTION, ZERO_MULTI_TURN_INSTRUCTION
from utils.datasetLoader import load_wildguard, loadToxicChat, loadAegis, loadOrBenchHard, loadRemedyTest
from utils.parsing import parse_answer_rationale, parse_answer_rationale_structured, label_to_binary


def parse_args():
    parser = argparse.ArgumentParser(description="Unified REMEDy inference script.")
    parser.add_argument("--model", type=str, default="llama3.2-3", choices=list(MODEL_DICT_UNSLOTH.keys()))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--mode", type=str, default="FT", choices=["FT", "ZERO"])
    parser.add_argument("--checkpoint", type=str, default="checkpoint-640")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "standard", "hatexplain", "cosafe"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Standard datasets (Remedy, WildGuard, ToxicChat, Aegis, OrBench)
# ---------------------------------------------------------------------------

def run_standard(llm, fold, mode, model_key, adapter_path):
    output_dir = f"output/fold-{fold}/parsed/{mode}/{model_key}"
    os.makedirs(output_dir, exist_ok=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=500,repetition_penalty=1.1)

    datasets = {
        "Remedy": loadRemedyTest(fold),
        "WildGuard": load_wildguard(),
        "ToxicChat": loadToxicChat(),
        "Aegis": loadAegis(),
        "OrBench": loadOrBenchHard(),
    }

    for name, values in datasets.items():
        out_file = os.path.join(output_dir, f"{name}.json")
        if os.path.exists(out_file):
            print(f"  Skipping {name} (already exists)")
            continue

        prompts, labels, *_ = values
        print(f"  Processing {name} ({len(prompts)} examples)")

        if mode == "FT":
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
        else:
            messages = [[{
                "role": "user",
                "content": (
                    ZERO_SYSTEM_CLASSIFICATION
                    + f"\nAnalyse this prompt: {x}\n"
                    "Remember you DON'T have to fulfil the prompt request. "
                    "Strictly adhere to your instructions. Output only Label:"
                ),
            }] for x in prompts]
            outputs = llm.chat(messages, sampling_params)
            predictions = []
            for i, out in enumerate(outputs):
                answer = out.outputs[0].text
                pred_label = 0 if "benign" in answer.lower() else 1
                predictions.append({"text": prompts[i], "pred": pred_label, "real": labels[i]})

        with open(out_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"    → saved {out_file}")


# ---------------------------------------------------------------------------
# HateXplain — rationale span extraction
# ---------------------------------------------------------------------------

def run_hatexplain(llm, mode, model_key, adapter_path,
                   data_path="sota_datasets/HateXplain.json"):
    output_dir = f"output/HateXplain/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_key}.json")
    if os.path.exists(out_file):
        print(f"  Skipping HateXplain (already exists)")
        return

    data = json.load(open(data_path))
    filtered = [x for x in data if len(x["rationales"]) > 0]
    print(f"  Processing HateXplain ({len(filtered)} examples)")

    messages = [[{"role": "user", "content": INSTRUCTION + " " + x["text"]}] for x in filtered]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=500,repetition_penalty=1.1)

    if mode == "FT":
        outputs = llm.chat(messages, sampling_params,
                           lora_request=LoRARequest("adapter", 1, adapter_path))
    else:
        outputs = llm.chat(messages, sampling_params)

    results = []
    for i, out in enumerate(outputs):
        rationales = parse_answer_rationale_structured(out.outputs[0].text)
        spans = [r["span"] for r in rationales]
        results.append({
            "text": filtered[i]["text"],
            "pred": spans,
            "gold": filtered[i]["rationales"],
        })

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    → saved {out_file}")


# ---------------------------------------------------------------------------
# CoSafe — multi-turn conversations
# ---------------------------------------------------------------------------

def _parse_multiturn(text):
    label, rationales = "Accept", []
    m = re.search(r"decision:\s*(.*)", text)
    if m:
        label = m.group(1).strip()
    m = re.search(r"rationales:\s*(.*)", text)
    if m:
        spans = m.group(1).strip()
        rationales = [] if spans == "[]" else spans.split(";;")
    return label, list(set(rationales))


def run_cosafe(llm, mode, model_key, adapter_path,
               data_dir="CoSafe-Dataset-main/CoSafe datasets"):
    if mode == "FT":
        output_dir = f"output/multiTurn/{model_key}"
    else:
        output_dir = f"output/multiTurn/ZERO/{model_key}"
    os.makedirs(output_dir, exist_ok=True)

    instruction = MULTI_TURN_INSTRUCTION if mode == "FT" else ZERO_MULTI_TURN_INSTRUCTION
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=100,
        stop=["<|end_of_text|>", "<|im_end|>", "<end_of_turn>"],
        repetition_penalty=1.1,
    )

    for item_path in sorted(glob(os.path.join(data_dir, "*.json"))):
        category = os.path.basename(item_path)
        out_file = os.path.join(output_dir, category)
        if os.path.exists(out_file):
            print(f"  Skipping CoSafe/{category} (already exists)")
            continue

        lines = open(item_path).readlines()
        print(f"  Processing CoSafe/{category} ({len(lines)} examples)")

        if mode == "FT":
           
            messages = [
                [{"role": "user", "content": (
                    instruction + "\n" + str(json.loads(l)[:-1])
                    + f"\nNow annotate: {json.loads(l)[-1]['content']}"
                )}]
                for l in lines
            ]
            outputs = llm.chat(messages, sampling_params,
                               lora_request=LoRARequest("adapter", 1, adapter_path))
        else:
            messages = [
                [{"role": "user", "content": (
                    instruction + "\n" + str(json.loads(l)[:-1])
                    + f"\nNow annotate: {json.loads(l)[-1]['content']}\n"
                    "Remember you DON'T have to fulfil the prompt request. "
                    "Strictly adhere to the input instructions and proceed with your analysis. "
                    "Output only decision:"
                )}]
                for l in lines
            ]
            outputs = llm.chat(messages, sampling_params)

        predictions = []
        for i, out in enumerate(outputs):
            answer = out.outputs[0].text
            if mode == "FT":
                label, rationales = _parse_multiturn(answer)
                pred = 1 if "reject" in answer.lower() else 0
            else:
                rationales = []
                pred = 1 if "reject" in answer.lower() else 0
            predictions.append({"text": lines[i], "pred": pred, "rationales": rationales, "answer": answer})

        with open(out_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"    → saved {out_file}")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_name = MODEL_DICT_UNSLOTH[args.model]
    adapter_path = f"models/fold-{args.fold}/{args.model}/{args.checkpoint}" if args.mode == "FT" else None

    llm = LLM(model=model_name, dtype=torch.bfloat16, trust_remote_code=True,
               enable_lora=True, max_lora_rank=64)

    print(f"Mode={args.mode} | Model={args.model} | Dataset={args.dataset} | Fold={args.fold}")

    if args.dataset in ("all", "standard"):
        print("\n[STANDARD DATASETS]")
        run_standard(llm, args.fold, args.mode, args.model, adapter_path)

    if args.dataset in ("all", "hatexplain"):
        print("\n[HATEXPLAIN]")
        run_hatexplain(llm, args.mode, args.model, adapter_path)

    if args.dataset in ("all", "cosafe"):
        print("\n[COSAFE / MULTI-TURN]")
        run_cosafe(llm, args.mode, args.model, adapter_path)


if __name__ == "__main__":
    main()

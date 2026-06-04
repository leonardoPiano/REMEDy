"""WildGuard baseline inference."""
import os
import json
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.datasetLoader import load_wildguard, loadToxicChat, loadAegis, loadOrBenchHard, loadRemedyTest
from .prompts.wildguard_prompt import instruction_format


MODEL_ID = "allenai/wildguard"
MODEL_NAME = "wildguard"


def _take_label(decision: str) -> int:
    return 1 if "yes" in decision.lower() else 0


def run(fold: int = 0, cuda_device: str = "0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    out_path = f"output/fold-{fold}/parsed/GUARD/{MODEL_NAME}"
    os.makedirs(out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, device_map="cuda:0")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda:0")

    def predict(prompt: str) -> int:
        model_input = instruction_format.format(prompt=prompt, response="")
        tokenized = tokenizer([model_input], return_tensors="pt", add_special_tokens=False).to("cuda:0")
        result = model.generate(**tokenized, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(result[0][len(tokenized["input_ids"][0]):], skip_special_tokens=True)
        return _take_label(answer.split("\n")[0])

    datasets = {
        "Remedy": loadRemedyTest(fold),
        "ToxicChat": loadToxicChat(),
        "Aegis": loadAegis(),
        "OrBench": loadOrBenchHard(),
    }

    for name, data in datasets.items():
        print(f"Processing {name}")
        prompts, labels, *_ = data
        results = [
            {"text": p, "pred": predict(p), "real": l}
            for p, l in tqdm(zip(prompts, labels), total=len(prompts))
        ]
        out_file = os.path.join(out_path, f"{name}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    args = parser.parse_args()
    run(fold=args.fold, cuda_device=args.cuda)

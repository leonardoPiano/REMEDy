"""GLiNER inference script — runs all span-based experiments.

Experiments:
  standard   All 3 folds × {gliner, nuner} on Remedy + OOD benchmarks
             → output/fold-{fold}/parsed/FT/{model}/{dataset}.json
  hatexplain  Fold-0 × {gliner, nuner} on HateXplain (rationale extraction)
             → output/HateXplain/FT/{model}.json

Classification rule:
  - no predictions OR all labels are "harmless"  →  BENIGN  (pred=0)
  - at least one non-harmless span predicted      →  MALICIOUS (pred=1)

Run from the project root:
  python span-based/inference.py                          # all models, all datasets
  python span-based/inference.py --model gliner           # gliner only
  python span-based/inference.py --dataset hatexplain     # HateXplain only
  python span-based/inference.py --model nuner --dataset standard
"""
import os
import sys
import json
import argparse
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Allow importing utils from the project root regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tqdm import tqdm
from gliner import GLiNER
from sklearn.metrics import accuracy_score, f1_score

from utils.datasetLoader import load_wildguard, loadToxicChat, loadAegis, loadOrBenchHard, loadRemedyTest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "gliner": "urchade/gliner_large-v2.1",
    "nuner": "numind/NuNerZero_span",
}

# Threshold for span detection on standard classification datasets
CLASSIFICATION_THRESHOLD = 0.30

# Threshold / labels used specifically for HateXplain
HATEXPLAIN_THRESHOLD = 0.10
HATEXPLAIN_LABELS = ["offensive language", "hatespeech"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_entity_labels() -> list[str]:
    schema = json.load(open("dataset/gold/REMEdy_schema.json"))
    return (
        list(schema["MALICIOUS"]["categories"].keys())
        + list(schema["TARGETS"]["categories"].keys())
        + ["harmless"]
    )


def load_model(model_name: str, fold: int) -> GLiNER:
    """Load the best checkpoint for a given model and fold."""
    pattern = f"span-based/models/fold-{fold}/{model_name}/checkpoint-*"
    checkpoints = sorted(glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
    ckpt = checkpoints[-1]
    print(f"  Loading checkpoint: {ckpt}")
    model = GLiNER.from_pretrained(ckpt, device_map="cuda:0").to("cuda:0")
    model.data_processor.config.max_len = 500
    return model


def classify(predictions: list) -> int:
    """Convert span predictions to a binary malicious/benign label."""
    if not predictions:
        return 0
    if all(p["label"] == "harmless" for p in predictions):
        return 0
    return 1


# ---------------------------------------------------------------------------
# Standard datasets (Remedy test + OOD benchmarks)
# ---------------------------------------------------------------------------

def run_standard(models: list[str], folds: list[int], entity_labels: list[str]):
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")

        # Load datasets once per fold (Remedy is fold-specific)
        datasets = {
            "Remedy": loadRemedyTest(fold),
            "WildGuard": load_wildguard(),
            "ToxicChat": loadToxicChat(),
            "Aegis": loadAegis(),
            "OrBench": loadOrBenchHard(),
        }

        for model_name in models:
            print(f"\n  Model: {model_name}")
            output_dir = f"output/fold-{fold}/parsed/FT/{model_name}"
            os.makedirs(output_dir, exist_ok=True)

            model = load_model(model_name, fold)

            for dataset_name, values in datasets.items():
                out_file = os.path.join(output_dir, f"{dataset_name}.json")
                if os.path.exists(out_file):
                    print(f"    Skipping {dataset_name} (already exists)")
                    continue

                prompts, labels, *_ = values
                print(f"    Processing {dataset_name} ({len(prompts)} examples)")

                predictions_out = []
                pred_labels = []

                for i, prompt in enumerate(tqdm(prompts, desc=f"      {dataset_name}")):
                    preds = model.predict_entities(
                        prompt, entity_labels,
                        flat_ner=False, threshold=CLASSIFICATION_THRESHOLD
                    )
                    pred_lbl = classify(preds)
                    pred_labels.append(pred_lbl)
                    predictions_out.append({
                        "text": prompt,
                        "pred": pred_lbl,
                        "real": labels[i],
                        "rationales": [f"{p['text']} ({p['label']})" for p in preds],
                    })

                with open(out_file, "w") as f:
                    json.dump(predictions_out, f, indent=2)

                acc = accuracy_score(labels, pred_labels)
                f1 = f1_score(labels, pred_labels, zero_division=0)
                print(f"    → saved {out_file}  ACC={acc:.3f}  F1={f1:.3f}")

            del model  # free GPU memory before next model


# ---------------------------------------------------------------------------
# HateXplain — rationale span extraction
# ---------------------------------------------------------------------------

def run_hatexplain(models: list[str], fold: int = 0,
                   data_path: str = "sota_datasets/HateXplain.json"):
    print(f"\n{'='*60}")
    print("HATEXPLAIN")

    data = json.load(open(data_path))
    filtered = [x for x in data if len(x["rationales"]) > 0]
    print(f"  Loaded {len(filtered)} examples with rationales")

    output_dir = "output/HateXplain/FT"
    os.makedirs(output_dir, exist_ok=True)

    for model_name in models:
        out_file = os.path.join(output_dir, f"{model_name}.json")
        if os.path.exists(out_file):
            print(f"  Skipping {model_name} (already exists)")
            continue

        model = load_model(model_name, fold)

        results = []
        for item in tqdm(filtered, desc=f"  {model_name}"):
            preds = model.predict_entities(
                item["text"], HATEXPLAIN_LABELS,
                flat_ner=True, threshold=HATEXPLAIN_THRESHOLD
            )
            results.append({
                "text": item["text"],
                "pred": [p["text"] for p in preds],
                "gold": item["rationales"],
            })

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → saved {out_file}")

        del model


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GLiNER inference for REMEDy experiments.")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all"] + list(MODEL_IDS.keys()),
                        help="Which model(s) to run (default: all)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "standard", "hatexplain"],
                        help="Which experiment(s) to run (default: all)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold for standard datasets (default: all three folds 0-2)")
    return parser.parse_args()


def main():
    args = parse_args()

    models = list(MODEL_IDS.keys()) if args.model == "all" else [args.model]
    folds = list(range(3)) if args.fold is None else [args.fold]
    entity_labels = load_entity_labels()

    print(f"Models:  {models}")
    print(f"Dataset: {args.dataset}")
    if args.dataset in ("all", "standard"):
        print(f"Folds:   {folds}")

    if args.dataset in ("all", "standard"):
        run_standard(models, folds, entity_labels)

    if args.dataset in ("all", "hatexplain"):
        # HateXplain uses fold-0 checkpoint by convention
        run_hatexplain(models, fold=folds[0])


if __name__ == "__main__":
    main()

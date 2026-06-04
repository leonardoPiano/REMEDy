"""Fine-tune a GLiNER-style span extraction model on the REMEDy dataset.

Supported models:
  gliner  urchade/gliner_large-v2.1
  nuner   numind/NuNerZero_span

Run from the project root:
  python span-based/train.py --model gliner --fold 0 --epochs 15
  python span-based/train.py --model nuner  --fold 1 --epochs 15
"""
import os
import sys
import json
import random
import argparse
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Allow importing utils from the project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
from gliner_utils.customEval import CustomEvalDataloaderCallback, custom_evaluation

random.seed(42)

MODEL_IDS = {
    "gliner": "urchade/gliner_large-v2.1",
    "nuner": "numind/NuNerZero_span",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def transform_example(item: dict, labels: list[str]) -> dict:
    """Convert one annotated example to the GLiNER token+span format."""
    tokens = [t.lower() for t in tokenize(item["text"])]
    spans = []
    for entity in item["rationales"]:
        entity_tokens = tokenize(entity["span"].lower())
        length = len(entity_tokens)
        for i in range(len(tokens) - length + 1):
            if tokens[i : i + length] == entity_tokens:
                spans.append([i, i + length - 1, entity["label"]])
                break
    return {"tokenized_text": tokens, "ner": spans, "label": labels}


def build_dataset(raw_data: list, labels: list[str]) -> list:
    return [
        transform_example({"text": item["text"], "rationales": item["rationales"]}, labels)
        for item in raw_data
    ]


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER on REMEDy.")
    parser.add_argument("--model", type=str, default="nuner", choices=list(MODEL_IDS.keys()))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load entity label taxonomy from schema
    schema = json.load(open("./dataset/REMEdy_schema.json"))
    labels = (
        list(schema["MALICIOUS"]["categories"].keys())
        + list(schema["TARGETS"]["categories"].keys())
        + ["harmless"]
    )

    # Load data
    train_raw = json.load(open(f"./dataset/fold-{args.fold}/train.json"))
    test_raw = json.load(open(f"./dataset/fold-{args.fold}/test.json"))
    random.shuffle(train_raw)

    train_dataset = build_dataset(train_raw, labels)
    test_dataset = build_dataset(test_raw, labels)

    # Load model
    model = GLiNER.from_pretrained(MODEL_IDS[args.model], device_map="cuda:0").to("cuda:0")

    # Build eval dataloader with extra fields needed by custom callback
    eval_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        return_tokens=True,
        return_entities=True,
        return_id_to_classes=True,
        prepare_labels=False,
        entity_types=labels,
    )
    eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_collator
    )
    eval_callback = CustomEvalDataloaderCallback(eval_loader, custom_evaluation)

    # Training collator (with label preparation)
    train_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    total_steps = args.epochs * (len(train_dataset) / args.batch_size)
    warmup_steps = max(1, int(total_steps * 0.1))

    training_args = TrainingArguments(
        output_dir=f"span-based/models/fold-{args.fold}/{args.model}",
        learning_rate=args.lr,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        save_strategy="steps",
        save_total_limit=2,
        metric_for_best_model="F1",
        greater_is_better=True,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        callbacks=[eval_callback],
        data_collator=train_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()

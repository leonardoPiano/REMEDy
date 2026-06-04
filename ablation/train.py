"""Ablation training script.

Variants:
  --ablation no_rationale   train with labels only (no rationale spans)
  --ablation no_harmless    train with rationales but exclude NEUTRAL (harmless) spans

Models are saved to:
  ablation/models/fold-{fold}/{ablation}/{model}/
"""
import os
import sys
import random
import json
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats

from utils.train_utils import MODEL_DICT
from prompts import NO_RATIONALE_INSTRUCTION,RATIONALE_INSTRUCTION

random.seed(42)



def parse_args():
    parser = argparse.ArgumentParser(description="Ablation training for REMEDy.")
    parser.add_argument("--model", type=str, default="llama3.2-3", choices=list(MODEL_DICT.keys()))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["no_rationale", "no_harmless"],
                        help="no_rationale: train with labels only; no_harmless: exclude neutral spans")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def create_ablation_dataset(dataset, ablation, neutrals):
    conv_data = []
    for item in dataset:
        text = item["text"]
        label = item["macro_label"]
        label = "MALICIOUS" if label == "MALICIOUS" else "BENIGN"

        if ablation == "no_rationale":
             prompt = NO_RATIONALE_INSTRUCTION + " " + text           
             response = f"label: {label}"
        else:
            # no_harmless: keep only malicious/target rationale spans
            prompt = RATIONALE_INSTRUCTION + " " + text
            rationales = [r for r in item["rationales"] if r["label"] not in neutrals]
            answer = ";; ".join([f"{r['span']} ({r['label']})" for r in rationales]) if rationales else "[]"
            response = f"global_label: {label}\nrationales: {answer}"

        conv_data.append([
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response},
        ])
    return {"conversations": conv_data}


def formatting_prompts_func(examples, tokenizer):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for convo in examples["conversations"]
    ]
    return {"text": texts}


def main():
    args = parse_args()
    fold = f"fold-{args.fold}"
    model_id = MODEL_DICT[args.model]

    schema = json.load(open("dataset/REMEdy_schema.json"))
    neutrals = set(schema["NEUTRAL"]["categories"].keys())

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        load_in_4bit=False,
        full_finetuning=False,
        token=HF_TOKEN,
    )

   

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    data_path = f"dataset/gold/{fold}/train.json"
    out_path = f"ablation/models/{fold}/{args.ablation}/{args.model}"

    train_dataset = json.load(open(data_path))
    random.shuffle(train_dataset)

    hf_train = Dataset.from_dict(create_ablation_dataset(train_dataset, args.ablation, neutrals))
    unsloth_train = standardize_data_formats(hf_train)
    dataset_train = unsloth_train.map(
        lambda ex: formatting_prompts_func(ex, tokenizer), batched=True
    )

    batch_size = 4
    grad_accumulation = 4
    total_steps = args.epochs * (len(dataset_train) / batch_size) / grad_accumulation
    warmup_steps = max(1, int(total_steps * 0.05))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        args=SFTConfig(
            output_dir=out_path,
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation,
            warmup_steps=warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_steps=500,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()

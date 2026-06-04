"""Fine-tune an LLM on the REMEDy dataset with LoRA via Unsloth + TRL."""
import os
import random
import json
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats
import torch
from datasets import Dataset

from trl import SFTTrainer, SFTConfig
from utils.train_utils import MODEL_DICT, create_conversation_dataset

random.seed(42)




def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on REMEDy.")
    parser.add_argument("--model", type=str, default="llama3.2-3", choices=list(MODEL_DICT.keys()))
    parser.add_argument("--fold", type=int, default=1, help="Cross-validation fold index.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


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

    data_path = f"dataset/{fold}/train.json"
    out_path = f"models/{fold}/{args.model}"

    train_dataset = json.load(open(data_path))
    random.shuffle(train_dataset)

    hf_train = Dataset.from_dict(create_conversation_dataset(train_dataset))
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
        processing_class=tokenizer,
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

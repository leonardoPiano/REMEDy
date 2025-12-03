import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")
HF_TOKEN=os.environ.get("HF_TOKEN")
from unsloth import FastLanguageModel
import torch
import json
from trl import SFTTrainer, SFTConfig
from glob import glob
import random
from datasets import Dataset
from unsloth.chat_templates import standardize_data_formats
from train_utils import *
import argparse

random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3.2-3")
parser.add_argument("--epochs", type=str, default=2)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
             for convo in convos]
    return {"text": texts, }

model_dict={"llama3.1-8":"meta-llama/Llama-3.1-8B-Instruct","llama3.2-3":"meta-llama/Llama-3.2-3B-Instruct"}

if __name__=="__main__":
    args = parser.parse_args()
    model_name = args.model
    EPOCHS = int(args.epochs)

    model_id=model_dict[model_name]
   
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048, 
        load_in_4bit=False,  
        full_finetuning=False,  
        token=HF_TOKEN  
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha=32, 
        lora_dropout=0,  
        bias="none",  
        
        use_gradient_checkpointing="unsloth",  
        random_state=3407,
        use_rslora=False,  
        loftq_config=None,  
    )



    data_path="dataset/gold/train.json"
    train_dataset = json.load(open(data_path))
    random.shuffle(train_dataset)
    conv_train = create_conversation_dataset(train_dataset)

    hf_train = Dataset.from_dict(conv_train)


    unsloth_train = standardize_data_formats(hf_train)




    dataset_train = unsloth_train.map(formatting_prompts_func, batched=True)
    dataset_size=len(dataset_train)
    batch_size=4
    grad_accumulation=4
    total_steps = EPOCHS * (dataset_size / batch_size) / grad_accumulation
    warmup_ratio=0.05
    warmup_steps = int(total_steps * warmup_ratio)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,     
        args=SFTConfig(
            output_dir=model_name,
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation,  
            warmup_steps=5,
            num_train_epochs=EPOCHS,  
            learning_rate=2e-4,  
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            logging_steps=500,
            save_strategy="epoch",
    
            report_to="none",  
        ),
    )

    trainer_stats = trainer.train()
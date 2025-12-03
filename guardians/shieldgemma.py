import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
from dotenv import load_dotenv
from utils.datasetLoader import *
from .prompts.shieldgemma_guide import guideline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import json

load_dotenv(dotenv_path="../.env")
HF_TOKEN=os.environ.get("HF_TOKEN")

def predict(prompt):
    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(chat, guideline=guideline, return_tensors="pt", return_dict=True).to(
        model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
   # Extract the logits for the Yes and No tokens
    vocab = tokenizer.get_vocab()
    selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]

   # Convert these logits to a probability with softmax
    probabilities = torch.softmax(selected_logits, dim=0)

   # Return probability of 'Yes'
    score = probabilities[0].item()

    return score>0.50

if __name__=="__main__":
     tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-9b")
     model = AutoModelForCausalLM.from_pretrained(
            "google/shieldgemma-9b",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

     datasets = {"Remedy": loadRemedyTest(), "WildGuard": (load_wildguard()), "ToxicChat": (loadToxicChat()),
               "Aegis": (loadAegis()),
               "OrBench": (loadOrBenchHard())}
    
     out_path="output/parsed/GUARD/shieldgemma/"
    
     if not os.path.exists(out_path):
        os.mkdir(out_path)   
     
     for name, values in datasets.items():
        predictions = []
        print("Processing", name)
        prompts, labels, *df = values

        for prompt,label in tqdm(zip(prompts,labels)):
            predicted=int(predict(prompt))
            predictions.append({"text": prompt, "pred": predicted, "real": label})
        
        with open(f"{out_path}/{name}.json", "w") as f:
            json.dump(predictions, f, indent=2)



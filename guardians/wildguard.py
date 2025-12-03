import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"
from dotenv import load_dotenv
from glob import glob
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.datasetLoader import *
from .prompts.wildguard_prompt import instruction_format
from tqdm import tqdm
load_dotenv(dotenv_path="../.env")
HF_TOKEN=os.environ.get("HF_TOKEN")

def take_label(decision):    
    if "yes" in decision:
        return 1
    return 0
    
def predict(prompt):
    model_input = instruction_format.format(prompt=prompt, response="")
    tokenized_input = tokenizer([model_input], return_tensors='pt', add_special_tokens=False).to("cuda:0")
    result = model.generate(**tokenized_input, max_new_tokens=32,pad_token_id=tokenizer.eos_token_id)
    answer=tokenizer.decode(result[0][len(tokenized_input['input_ids'][0]):], skip_special_tokens=True)
    label=take_label(answer.split("\n")[0])
    return label

if __name__=="__main__":
    model_id = "allenai/wildguard"
    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="cuda:0")
    model = AutoModelForCausalLM.from_pretrained(model_id,device_map="cuda:0")


    out_path="output/parsed/GUARD/wildguard/"
    if not os.path.exists(out_path):
          os.mkdir(out_path)

    datasets = {"Remedy": loadRemedyTest(), "ToxicChat": (loadToxicChat()),
              "Aegis": (loadAegis()),
              "OrBench": (loadOrBenchHard())}
    
    for name, data in datasets.items():
        results=[]
        print("Processing",name)    
        prompts, labels,*df = data
        for prompt,real in tqdm(zip(prompts,labels)):
            pred=predict(prompt)
            results.append({"text":prompt,"pred":pred,"real":real})
        with open(f"{out_path}/{name}.json","w") as f:
            json.dump(results,f,indent=2)
    


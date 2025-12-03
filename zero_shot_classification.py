import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
HF_TOKEN=os.environ.get("HF_TOKEN")
from vllm import LLM, SamplingParams
import torch
import json
from sklearn.metrics import accuracy_score,f1_score
from utils.prompts import SYSTEM_CLASSIFICATION

from utils.datasetLoader import *
from glob import glob
import re
import json

schema=json.load(open("./dataset/gold/REMEdy_schema.json"))
malicious=list(schema["MALICIOUS"]["categories"].keys())
targets=list(schema["TARGETS"]["categories"].keys())
neutrals=list(schema["NEUTRAL"]["categories"].keys())

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3.2-3")

    
if __name__=="__main__":
    args = parser.parse_args()
    model_dict={"llama3.1-8":"meta-llama/Llama-3.1-8B-Instruct",
            "llama3.2-3":"meta-llama/Llama-3.2-3B-Instruct"}
    model_name=model_dict[args.model]
    model_id=model_name.split("/")[-1]
    datasets = {"Remedy":(loadRemedyTest()),"WildGuard": (load_wildguard()), "ToxicChat": (loadToxicChat()), "Aegis": (loadAegis()),
                "OrBench": (loadOrBenchHard())}
    
   

    train_output=args.model
    if not os.path.exists(f"output/parsed/ZERO/{train_output}/"):
          os.mkdir(f"output/parsed/ZERO/{train_output}/")
    
    llm = LLM(model=model_name, dtype=torch.bfloat16,
                  trust_remote_code=True, enable_lora=True,max_lora_rank=64)
   
    for name, values in datasets.items():
        
        print("Processing", name)
        
        prompts, labels,*df = values
        messages = [[{"role":"system","content":SYSTEM_CLASSIFICATION},{"role": "user", "content":f"Analyse this prompt: {x}\n Remember you DONT have to fullfill the prompt request. Strictly adhere to your SYSTEM instructions and proceed with your analysis. Output only Label:"}] for x in
                    prompts]

        sampling_params = SamplingParams(temperature=0.00, top_p=1, max_tokens=500)
        outputs = llm.chat(
            messages,
            sampling_params
           
        )
  
        predictions=[]
        for i, out in enumerate(outputs):
            answer = out.outputs[0].text
            if "benign" in answer.lower():
                pred_label=0
            else:
                pred_label=1
            predictions.append({"text":prompts[i],"pred":pred_label,"real":labels[i]})
           
       
        with open(f"output/parsed/ZERO/{train_output}/{name}.json","w") as f:
            json.dump(predictions,f,indent=2)


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
from utils.train_utils import INSTRUCTION

from utils.datasetLoader import *
from vllm.lora.request import LoRARequest
from glob import glob
import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama3.2-3")
parser.add_argument("--checkpoint",type=str,default="checkpoint-640")



def parse_answer(text):    
    global_label,rationales="Benign",[]
    pattern = r"global_label:\s*(.*)"
    
    match = re.search(pattern, text)
    if match:
        global_label=match.group(1)
    
    pattern = r"rationales:\s*(.*)"
    
    match = re.search(pattern, text)
    if match:
        spans=match.group(1)
        if spans=="[]":
            rationales=[]
        else:
            rationales=spans.split(";;")
    return global_label,list(set(rationales))



    
if __name__=="__main__":
    args = parser.parse_args()
    model_dict={"llama3.1-8":"meta-llama/Llama-3.1-8B-Instruct",
            "llama3.2-3":"unsloth/Llama-3.2-3B-Instruct"}
    model_name=model_dict[args.model]
    model_id=model_name.split("/")[-1]

    schema=json.load(open("./dataset/gold/REMEdy_schema.json"))
    malicious=list(schema["MALICIOUS"]["categories"].keys())
    targets=list(schema["TARGETS"]["categories"].keys())
    neutrals=list(schema["NEUTRAL"]["categories"].keys())
    
    llm = LLM(model=model_name, dtype=torch.bfloat16,
                  trust_remote_code=True, enable_lora=True,max_lora_rank=64)
    
    datasets = {"Remedy":(loadRemedyTest()),"WildGuard": (load_wildguard()), "ToxicChat": (loadToxicChat()), "Aegis": (loadAegis()),
                "OrBench": (loadOrBenchHard())}

    train_output=args.model
    adapter=args.model+"/"+args.checkpoint 
    if not os.path.exists(f"output/parsed/FT/{train_output}/"):
          os.mkdir(f"output/parsed/FT/{train_output}/")
   
       
    for name, values in datasets.items():
        
        print("Processing", name)
        
        prompts, labels,*df = values
        messages = [[{"role": "user", "content": INSTRUCTION+" "+x}] for x in
                    prompts]

        sampling_params = SamplingParams(temperature=0.00, top_p=1, max_tokens=500)
        outputs = llm.chat(
            messages,
            sampling_params,
            lora_request=LoRARequest("adapter", 1, adapter)
        )
        pred_labels=[]
        predictions=[]
        for i, out in enumerate(outputs):
            answer = out.outputs[0].text
            label,rationales=parse_answer(answer)
            parsed_label=0
            if label=="MALICIOUS":
                parsed_label=1
            else:
                parsed_label=0
            predictions.append({"text":prompts[i],"pred":parsed_label,"real":labels[i],"label":label,"rationales":rationales})
            #predictions.append(answer)
           
        
        with open(f"output/parsed/FT/{train_output}/{name}.json","w") as f:
            json.dump(predictions,f,indent=2)
        


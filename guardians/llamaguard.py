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

pattern = re.compile(r'^(.*?)\s*\(([^()]*)\)\s*$')
import json

def parse_answer(answer):
    if "unsafe" in answer:
        return 1
    else:
        return 0



if __name__=="__main__":
    sampling_params = SamplingParams(temperature=0.00, top_p=1, max_tokens=500)
    datasets = {"Remedy": loadRemedyTest(), "WildGuard": (load_wildguard()), "ToxicChat": (loadToxicChat()),
                "Aegis": (loadAegis()),
                "OrBench": (loadOrBenchHard())}

    model_name="llamaguard"
    model_id="meta-llama/Llama-Guard-3-8B"
    out_path=f"output/parsed/GUARD/{model_name}/"
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)   
          
    llm = LLM(model=model_id, dtype=torch.bfloat16,
              trust_remote_code=True)

    for name, values in datasets.items():
        predictions = []
        print("Processing", name)
        
        prompts, labels,*df = values
        messages = [[{"role": "user", "content": x}] for x in
                    prompts]

        outputs = llm.chat(
            messages,
            sampling_params
        )
        for i, out in enumerate(outputs):
            answer = out.outputs[0].text
            predictions.append({"text": prompts[i], "pred": parse_answer(answer),"real":labels[i]})
        
        
        
        with open(f"{out_path}/{name}.json","w") as f:
            json.dump(predictions,f,indent=2)

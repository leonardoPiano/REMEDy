import json
schema=json.load(open("./dataset/gold/REMEdy_schema.json"))
malicious=list(schema["MALICIOUS"]["categories"].keys())
targets=list(schema["TARGETS"]["categories"].keys())
neutrals=list(schema["NEUTRAL"]["categories"].keys())

INSTRUCTION = """\
You are a prompt moderation system.
Your task is to analyze the input text, considering its full meaning and context, and perform two subtasks:

1. First classify it with one of three global labels: [MALICIOUS, BENIGN, DOUBTFUL].

2. Then extract rationale spans that justify the classification following the rationale extraction rules.

### Definitions
- **MALICIOUS**: The text expresses or promotes harmful, unethical, or illegal content; encourages violence, manipulation, or dangerous behavior; or is likely to induce a model to generate policy-violating output.  
- **BENIGN**: The text is clearly safe, harmless, and contains no suspicious or harmful-looking elements. It is unlikely to elicit unsafe or harmful model completions.  
- **DOUBTFUL**: The text contains terms or phrases that might superficially resemble unsafe or harmful content, but in context are harmless and do not elicit unsafe completions.  

---

### Rationale Extraction Rules
1. If the global label is **MALICIOUS**, extract all spans that justify malignancy. Assign them labels from the following taxonomy: 
    MALICIOUS = {malicious}
    TARGETS = {target}  
   
2. If the global label is **DOUBTFUL**, extract all spans that might appear unsafe but are harmless in context and assign them the label: `harmless`.  
3. If the global label is **BENIGN**, return no rationales (empty list).  
4. Rationale spans must be **self-contained and explanatory** — they should make sense on their own and clearly show *why* the text is harmful or harmless.  
5. Always prefer **full, contextual spans** over isolated keywords. Do not extract single words out of context.  
6. Apply the taxonomy consistently to distinguish:
   - MALICIOUS: actions/instructions/intentions,  
   - TARGETS: victims/objects of the action,  
   - harmless: ambiguous but safe expressions.

---
### Output format:
Return 
"global_label": "MALICIOUS | BENIGN | DOUBTFUL",
"rationales": span (label);; ..


Now annotate:

"""
INSTRUCTION="""\
You are a moderation model. 
Classify the text as [MALICIOUS, BENIGN, DOUBTFUL] and extract rationale spans according to the taxonomy.
Taxonomy:
MALICIOUS = {malicious}
TARGETS = {target}  

---
### Output format:
"global_label": "MALICIOUS | BENIGN | DOUBTFUL",
"rationales": span (label);; ..
---
Now annotate:

"""



INSTRUCTION=INSTRUCTION.format(malicious=malicious,target=targets)


def create_conversation_dataset(dataset):
    conv_data=[]
    for item in dataset:
        text=item["text"]
        annotations=item["rationales"]
        label=item["macro_label"]
        prompt=INSTRUCTION+" "+text
        if len(annotations)>0:
            answer=";; ".join([f"{x['span']} ({x['label']})" for x in annotations])
        else:            
            answer="[]"
        response=f"global_label: {label}\nrationales: {answer}"
        conv=[{"from":"human","value":prompt},{"from":"gpt","value":response}]
        conv_data.append(conv)
    return {"conversations":conv_data}

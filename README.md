# REMEDy: Rationale Extraction for Moderation and Explainability of Dialogue Prompts

## Abstract

The wide adoption of conversational AI systems necessitates urgent and interpretable safety moderation, especially given that Large Language Models (LLMs) continue to exhibit vulnerabilities despite alignment efforts, posing significant risks to individual users, organisations, and society.

The ideal AI safety moderation system must be transparent and structurally interpretable. However, current moderation approaches typically rely on coarse classifications that offer limited interpretability and fail to capture the nuanced intent and contextual dependencies present in real-world user inputs. To advance moderation beyond these coarse labels, we present **REMEDy**, a novel dataset specifically built for extracting fine-grained rationales from user prompts. REMEDy features span-level annotations covering a broad taxonomy of safety-relevant categories, allowing for overlapping and nested textual spans to reflect complex prompt structures.

Using REMEDy, we fine-tune multiple LLMs and span-based models and evaluate their performance across two tasks: **(i) rationale extraction**, assessing their ability to accurately localise and classify harmful or ambiguous content; and **(ii) prompt moderation**, measuring improvements over state-of-the-art safety detectors. Our experiments demonstrate that REMEDy-trained models achieve competitive or superior moderation outcomes while simultaneously providing structured, human-readable rationales. REMEDy thus offers a valuable resource for developing safer, more transparent, and context-sensitive moderation systems.

---

## Dataset Description

REMEDy is designed to support fine-grained, rationale-aware LLM safety moderation. Each example is annotated at the span level with a custom taxonomy divided into three macro-groups:

- **Malicious spans** — explicit mentions of harmful or policy-violating actions, behaviors, or entities.
- **Target spans** — the recipient or object of the malicious action.
- **Neutral spans** — content that may superficially resemble harmful material but is benign in context.

### Taxonomy

| Group | Label | Description |
|---|---|---|
| **Malicious** | offensive language | Vulgar expressions or language intended to offend. |
| **Malicious** | hateful or discriminatory act | Behaviour motivated by hatred or bias based on personal traits. |
| **Malicious** | self harm or suicidal thoughts | Mentions of intentional self-injury or suicide ideation. |
| **Malicious** | digital crime and violation | Illegal or unauthorised digital activity (e.g., hacking, data theft). |
| **Malicious** | copyright violation | Unauthorised use, reproduction, or distribution of copyrighted material. |
| **Malicious** | harmful instrument | Objects or tools used to cause harm, destruction, or intimidation. |
| **Malicious** | crime act | Acts violating criminal law. |
| **Malicious** | violent act | Physical aggression toward people, animals, or property. |
| **Malicious** | manipulative act | Covert attempts to deceive or unduly influence others. |
| **Malicious** | sexual content | Material depicting or alluding to sexual acts or nudity. |
| **Malicious** | minor abuse | Physical, emotional, or sexual harm or exploitation of children. |
| **Malicious** | substance abuse | Abuse of psychoactive substances (alcohol, drugs). |
| **Malicious** | illicit drug | Mentions of substances banned by law. |
| **Malicious** | drug trafficking | Illegal production, sale, or distribution of controlled substances. |
| **Target** | target living being | A person or animal identified as the recipient of malicious action. |
| **Target** | target organisation | An institution, business, or group targeted by an action. |
| **Target** | target location | A geographical place referenced as the object of action. |
| **Target** | target physical object | A tangible item serving as the target. |
| **Target** | target digital entity | A digital or fictional construct referenced as the target. |
| **Target** | target abstract | An intangible entity such as ideas or ideologies. |
| **Neutral** | harmless | Content that appears harmful but is contextually benign. |

---

## Results

All metrics are **F1** unless otherwise noted. FT results are averaged over **3-fold cross-validation** (mean ± std). Zero-Shot and Guardian results are evaluated on fold-0.

### Prompt Classification — (F1)

| Model | Remedy | ToxicChat | AEGIS | OR-Bench-hard |
|---------|--------:|--------:|--------:|--------:|
| *Guardian models* | | | | |
| LlamaGuard3-8b | 0.83 | 0.54 | 0.72 | 0.81 |
| ShieldGemma-9b | 0.79 | 0.68 | 0.76 | 0.74 |
| WildGuard-7b | 0.96 | 0.70 | 0.89 | 0.29 |
| QwenDuoGuard-0.5b | 0.83 | 0.70 | 0.78 | 0.77 |
| QwenDuoGuard-1.5b | 0.87 | 0.66 | 0.80 | 0.78 |
| LlamaDuoGuard-1b | 0.87 | 0.65 | 0.83 | 0.77 |
| *Zero-shot* | | | | |
| Llama3.2-3b | 0.86 | 0.60 | 0.73 | 0.81 |
| Llama3.1-8b | 0.87 | 0.58 | 0.77 | 0.77 |
| Gemma2-9b | 0.90 | 0.68 | 0.85 | 0.15 |
| Mistral-7b-v0.3 | 0.88 | 0.69 | 0.78 | 0.75 |
| Llama3.3-70b | 0.92 | 0.72 | 0.84 | 0.50 |
| GPT-4o-mini | 0.90 | 0.68 | 0.86 | 0.13 |
| *Fine-tuned* | | | | |
| Llama3.2-3b | 0.96 | 0.75 | 0.83 | **0.82** |
| Llama3.1-8b | 0.96 | 0.72 | 0.86 | 0.71 |
| Gemma2-9b | **0.98** | **0.78** | **0.87** | 0.75 |
| Mistral-7b-v0.3 | 0.97 | **0.78** | 0.86 | 0.70 |
| *Span classifiers* | | | | |
| NuNER-Zero-span | 0.94 | 0.70 | 0.84 | 0.67 |
| GliNER-large | 0.93 | 0.74 | 0.84 | 0.71 |




### WildGuard Out-of-Domain (F1)
| Model | Overall F1 | Vanilla F1 | Adversarial F1 |
|---------|--------:|--------:|--------:|
| *Guardian models* | | | |
| LlamaGuard3-8b | 0.77 | 0.87 | 0.62 |
| ShieldGemma-9b | 0.56 | 0.66 | 0.41 |
| QwenDuoGuard-0.5b | 0.77 | 0.78 | 0.75 |
| QwenDuoGuard-1.5b | 0.78 | 0.80 | 0.74 |
| LlamaDuoGuard-1b | 0.82 | 0.84 | 0.79 |
| *Zero-shot* | | | |
| Llama3.2-3b | 0.68 | 0.79 | 0.52 |
| Llama3.1-8b | 0.71 | 0.84 | 0.50 |
| Gemma2-9b | 0.84 | 0.91 | 0.77 |
| Mistral-7b-v0.3 | 0.78 | 0.87 | 0.63 |
| Llama3.3-70b | **0.86** | 0.91 | 0.80 |
| GPT-4o-mini | **0.86** | 0.90 | **0.82** |
| *Fine-tuned* | | | |
| Llama3.2-3b | 0.81 | 0.89 | 0.71 |
| Llama3.1-8b | 0.85 | **0.93** | 0.75 |
| Gemma2-9b | 0.86 | 0.92 | 0.79 |
| Mistral-7b-v0.3 | 0.83 | 0.90 | 0.74 |
| *Span classifiers* | | | |
| NuNER-Zero-span | 0.69 | 0.82 | 0.49 |
| GliNER-large | 0.72 | 0.83 | 0.56 |


### Multi-Turn Moderation — CoSafe (Accuracy)

CoSafe contains exclusively harmful multi-turn conversations; precision measures the rejection rate per category.

| Category | gemma2 FT |
|---|---|
| drug\_abuse / weapons / banned\_substance | 0.94 |
| financial\_crime / property\_crime / theft | 0.93 |
| discrimination / stereotype / injustice | 0.93 |
| sexually\_explicit / adult\_content | 0.92 |
| privacy\_violation | 0.92 |
| child\_abuse | 0.92 |
| self\_harm | 0.91 |
| hate\_speech / offensive\_language | 0.90 |
| non\_violent\_unethical\_behavior | 0.89 |
| violence / aiding\_and\_abetting / incitement | 0.88 |
| controversial\_topics / politics | 0.88 |
| terrorism / organized\_crime | 0.87 |
| animal\_abuse | 0.80 |
| misinformation (ethics / laws / safety) | 0.77 |
| **Overall** | **0.89** |

### Ablation Study (fold-0, F1)

Impact of rationale annotation on classification performance.

| Model | Variant | REMEDy | Aegis | ToxicChat | WildGuard Overall |
|---|---|---|---|---|---|
| **gemma2** | Full (w/ rationales) | **0.979** | 0.872 | **0.759** | 0.843 |
| **gemma2** | No Harmless | 0.974 | 0.872 | 0.730 | **0.867** |
| **gemma2** | No Rationale | 0.962 | **0.884** | 0.720 | 0.856 |
| mistral | Full (w/ rationales) | **0.969** | 0.860 | **0.749** | 0.843 |
| mistral | No Harmless | 0.957 | 0.827 | 0.767 | 0.839 |
| mistral | No Rationale | 0.963 | **0.866** | 0.736 | **0.859** |
| llama3.2-3 | Full (w/ rationales) | **0.964** | 0.823 | 0.718 | 0.810 |
| llama3.2-3 | No Harmless | 0.951 | 0.829 | **0.753** | 0.804 |
| llama3.2-3 | No Rationale | 0.951 | **0.840** | 0.738 | **0.834** |

> **Full** = trained with complete rationale annotation (malicious + target + neutral spans).  
> **No Harmless** = rationales excluding neutral (harmless) spans.  
> **No Rationale** = label-only supervision, no span annotation.

---

## Repository Structure

```
REMEDy/
│
├── dataset/gold/
│   ├── fold-{0,1,2}/train.json      # 3-fold cross-validation splits
│   ├── fold-{0,1,2}/test.json
│   └── REMEdy_schema.json           # Full taxonomy definition
│
├── utils/
│   ├── train_utils.py               # INSTRUCTION prompt, model dicts, conversation builder
│   ├── prompts.py                   # Zero-shot and multi-turn prompt templates
│   ├── datasetLoader.py             # Loaders for all benchmark datasets
│   └── parsing.py                   # Output parsing helpers
│
├── train.py                         # Fine-tune LLMs with LoRA (Unsloth + TRL)
├── inference.py                     # Unified inference: FT/ZERO × standard/HateXplain/CoSafe
│
├── span-based/
│   ├── train.py                     # Fine-tune GLiNER/NuNer span extraction models
│   ├── inference.py                 # Span-based inference (all models, all datasets)
│   └── gliner_utils/                # Custom eval callback and evaluator
│
├── ablation/
│   ├── train.py                     # Ablation training: no_rationale / no_harmless
│   ├── inference.py                 # Ablation inference → ablation/output/
│   ├── PromptClassification.ipynb   # Ablation evaluation notebook
│   └── output/                      # Ablation inference outputs (preserved)
│
├── guardians/
│   ├── llamaguard.py                # Meta LlamaGuard-3-8B
│   ├── wildguard.py                 # AllenAI WildGuard
│   ├── shieldgemma.py               # Google ShieldGemma-9B
│   └── duoguard.py                  # DuoGuard (0.5B / 1.5B / LLaMA variants)
│
├── evaluation/
│   ├── PromptClassification.ipynb   # Classification metrics (FT/ZERO/GUARD/HateXplain/CoSafe)
│   ├── SpanEvaluation.ipynb         # Rationale span-matching metrics
│   └── eval_metrics.py              # Soft span-match F1 (Jaccard-based)
│
├── openRouter/                      # API-based inference notebooks (GPT-4o-mini, Llama-70B)
│
├── sota_datasets/
│   └── HateXplain.json              # HateXplain rationale extraction benchmark
│
├── output/                          # All inference outputs (do not delete)
│   ├── fold-{0,1,2}/parsed/
│   │   ├── FT/{model}/              # Fine-tuned LLM outputs
│   │   ├── ZERO/{model}/            # Zero-shot outputs
│   │   └── GUARD/{guardian}/        # Guardian baseline outputs
│   ├── HateXplain/FT|ZERO/{model}.json
│   └── multiTurn/{model}/ and ZERO/{model}/
│
└── scripts/
    ├── train.sh                     # Master: all training experiments
    ├── train_ft.sh                  # Fine-tune LLMs (4 models × 3 folds)
    ├── train_span.sh                # Fine-tune GLiNER models (2 models × 3 folds)
    ├── train_ablation.sh            # Ablation training (fold-0)
    ├── inference.sh                 # Master: all inference experiments
    ├── inference_ft.sh              # FT inference (standard + HateXplain + CoSafe)
    ├── inference_zero.sh            # Zero-shot inference
    ├── inference_span.sh            # Span-based inference
    ├── inference_ablation.sh        # Ablation inference
    └── guardians.sh                 # Guardian baseline inference
```

---

## Installation

```bash
git clone https://github.com/your-org/REMEDy.git
cd REMEDy

# Environment for LLM experiments (train + inference)
conda create -n remedy python=3.11
conda activate remedy
pip install unsloth trl vllm python-dotenv datasets scikit-learn pandas

# Separate environment for span-based models (GLiNER conflicts with vLLM)
conda create -n remedy-span python=3.11
conda activate remedy-span
pip install gliner scikit-learn datasets pandas

# HuggingFace token (required for gated models)
echo "HF_TOKEN=hf_your_token_here" > .env
```

---

## Reproducing the Experiments

All scripts are run from the **project root**. Every inference step skips outputs that already exist.

### Step 1 — Training

```bash
# All training in one command
bash scripts/train.sh

# Or individually:
bash scripts/train_ft.sh              # LLM fine-tuning (4 models × 3 folds)
bash scripts/train_span.sh            # GLiNER/NuNer (2 models × 3 folds)
bash scripts/train_ablation.sh        # Ablation variants (fold-0)
```

Single run examples:
```bash
python train.py --model gemma2 --fold 0 --epochs 2
python span-based/train.py --model nuner --fold 1 --epochs 15
python ablation/train.py --model llama3.2-3 --fold 0 --ablation no_rationale
```

### Step 2 — Inference

```bash
# All inference in one command (pass custom checkpoint if needed)
bash scripts/inference.sh                    # default: checkpoint-640
bash scripts/inference.sh checkpoint-960     # custom checkpoint

# Or individually:
bash scripts/inference_ft.sh                 # FT: standard + HateXplain + CoSafe
bash scripts/inference_zero.sh               # Zero-shot: all datasets, fold-0
bash scripts/inference_span.sh               # Span-based: all folds + HateXplain
bash scripts/inference_ablation.sh           # Ablation: fold-0
bash scripts/guardians.sh                    # Guardian baselines: all folds
bash scripts/guardians.sh 1                  # Run guardians on GPU 1
```

Single run examples:
```bash
# FT inference — all datasets
python inference.py --model gemma2 --fold 0 --mode FT --dataset all

# Zero-shot — standard benchmarks only
python inference.py --model mistral --fold 0 --mode ZERO --dataset standard

# Span-based — single model, all folds
python span-based/inference.py --model gliner --dataset all

# Ablation
python ablation/inference.py --model gemma2 --fold 0 --ablation no_harmless

# Guardian
python -m guardians.duoguard --all --fold 0
```

### Step 3 — Evaluation

Open the relevant notebook from the `evaluation/` folder in JupyterLab:

| Notebook | What it evaluates |
|---|---|
| `evaluation/PromptClassification.ipynb` | ACC & F1 for FT / Zero-Shot / Guardians / HateXplain / CoSafe |
| `evaluation/SpanEvaluation.ipynb` | Soft span F1 (Jaccard-based) for rationale extraction |
| `ablation/PromptClassification.ipynb` | Ablation study: full vs no-harmless vs no-rationale |

---

## Disclaimer ⚠️

The REMEDy dataset may include content that is **offensive or emotionally distressing**. Topics covered include, but are not limited to, **discriminatory language**, references to **abuse**, **violence**, **self-harm**, **drugs**, and other **sensitive material**.

Please engage with the dataset only within your own personal comfort and risk tolerance. The material is provided solely for **research purposes**, particularly for work aimed at reducing harmful model behaviour. **The views expressed in the dataset do not represent the views of any organisations or authors involved in the project.**

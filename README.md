# REMEDy: A Dataset for Rationale Extraction and Span-Based Moderation of Dialogue Prompts
# ABSTRACT
The wide adoption of conversational AI systems necessitates urgent and interpretable safety moderation, especially given that Large Language Models (LLMs) continue to exhibit vulnerabilities despite alignment efforts, posing significant risks to individual users, organisations, and society.
The ideal AI safety moderation system must be transparent and structurally interpretable. However, current moderation approaches typically rely on coarse classifications that offer limited interpretability and fail to capture the nuanced intent and contextual dependencies present in real-world user inputs. To advance moderation beyond these coarse labels, we present REMEDy, a novel dataset specifically built for extracting fine-grained rationales from user prompts. REMEDy features span-level annotations covering a broad taxonomy of safety-relevant categories, allowing for overlapping and nested textual spans to reflect complex prompt structures.
Using REMEDy, we fine-tune multiple LLMs and evaluate their performance across two tasks: (i) rationale extraction, assessing their ability to accurately localise and classify harmful or ambiguous content; and (ii) prompt moderation, measuring improvements over state-of-the-art safety detectors. Our experiments demonstrate that REMEDy-trained models achieve competitive or superior moderation outcomes while simultaneously providing structured, human-readable rationales. REMEDy thus offers a valuable resource for developing safer, more transparent, and context-sensitive moderation systems.

# REMEDY DATASET DESCRIPTION
The REMEDy dataset has been designed to support fine-grained, rationale-aware LLM safety moderation systems.

To enable span-level rationale extraction, we designed a custom taxonomy.
The Taxonomy is divided in three macro-groups: 
- Malicious spans: Explicit mentions of harmful or policy-violating actions, behav-
iors, or entities.
-  Target spans: The recipient or object of the malicious action.
-  Neutral spans: Content that may superficially resemble harmful material but is
benign in context, included to capture ambiguity and borderline cases.
Below the full taxonomy



| **Group**   | **Label**                      | **Description** |
|-------------|--------------------------------|-----------------|
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

## Folder Description :open_file_folder:	
~~~

./dataset/gold          --> Contains the dataset related files (train, test and taxonomy).
./guardians             --> Contains the codes for run sota guardians (LLamaGuard, WildGuard and ShieldGemma)
./OpenRouter 	        --> Contains the codes for running the experiment (Prompt classification and Rationale Extraction) with OpenRouter API
./evaluation            --> Contains the evaluation code
./output/parsed         --> Contains all the output from all the runned experiments

~~~
# Disclaimer ⚠️
The REMEDy dataset may include content that is **offensive or emotionally distressing**. Topics covered include, but are not limited to, **discriminatory language**, references to **abuse**, **violence**, **self-harm**, **drugs**, and other **sensitive material**.

Please engage with the dataset only within your own personal comfort and risk tolerance. The material is provided solely for **research purposes**, particularly for work aimed at reducing harmful model behavior. **The views expressed in the dataset do not represent the views of any organizations or authors involved in the project.**









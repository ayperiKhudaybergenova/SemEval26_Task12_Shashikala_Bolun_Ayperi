# Task-12

The Abductive Event Reasoning (AER) aiming to evaluate large language models' ability to identify the most plausible direct cause of a real-world event based on textual evidence


https://sites.google.com/view/semeval2026-task12/


 # Releated papers for the task

https://arxiv.org/pdf/2305.16646

https://aclanthology.org/2025.acl-long.1269.pdf?utm_source=chatgpt.com

https://arxiv.org/html/2502.03671v1#S3


   
 
## Timeline :
![schedule]( https://github.com/ayperiKhudaybergenova/Task-12/blob/main/Main%20Schedule.png)


# Identified Steps 
## Step 1: Load and Understand the Data
   - Get the dataset from GitHub:
   - https://github.com/sooo66/semeval2026-task12-dataset
   

## Step 2: Preprocess the Data
   - preporocess data | if need format the data .
   
## Step 3: Choose a Baseline Model
   -   Llama | Qwen(free model) | DeBERTa
   -   GPU ? try bwHPC?


Accuracy is correct score/number of questions. 


**Evaluation Dataset: val_data**


| Peak Accuracy | Qwen | Llama | DeBERTa | DeBERTa *Properly done with new metric | Qwen *ToT with new metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Sample_data | 77.11% | 61.67% | 70 | 62.2% | 51.74% |


Details:

Qwen: Zero-shot,English prompt is the best, model: [Qwen3-30B-A3B-Thinking-2507-GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF)


**Evaluation Dataset: sample_data**


| Peak Accuracy | Qwen | Llama | DeBERTa |
| :--- | :--- | :--- | :--- |
| Sample_data | 71.25% | 60.00% |͌ 60% |


Details:

Qwen: Zero-shot,English prompt is the best, forgot to mention there're distractor docs in system prompt, model: [Qwen3-30B-A3B-Thinking-2507-GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF)
| Accuracy | Qwen |
| :--- | :--- |
| Zero-shot, English prompt | 71.25% |
| Zero-shot, Chinese prompt | 66.25% |
| Few-shot, English prompt | 62.29% |
|Multi-Expert, English prompt | 60% |


DeBERTa: As DeBERTa wasn’t originally trained with a multiple-choice head, we treated each option as a separate "sentence pair" with the question and encode (question + option) separately and pick the option with highest score.

## Step 4: Prepare Model Input
   - Create prompt
     
## Step 5: Train the Model
   - Using train split
     
# Step 6: Evaluate the Model (Official Metric just Changed Recently)

### Old metric(score without * on this readme)
   - Get predited asnwers
   - Compare with correct answers (Full match → 1.0 | Partial match → 0.5 | Wrong → 0.0)

### New metric(score with * on this readme)

System performance is evaluated at the instance level using an exact and partial matching scheme over the predicted answer options.

Let G denote the set of gold-standard correct options for an instance, and P denote the set of options predicted by the system.

Each instance is scored as follows:

1.0 (Full Match): if P = G

0.5 (Partial Match): if P is a non-empty proper subset of G (i.e., the prediction covers at least one correct option and contains no incorrect options)

0.0 (Incorrect): otherwise, including cases where the prediction contains any incorrect option or is empty

The final system score is computed as the average score across all evaluation instances.

Predictions must contain only valid option labels (A–D). Invalid labels result in a score of 0 for the corresponding instance.
     
## Step 7: Addional Evaluation (Extra)
   - Multi-Step Reasoning
   - Evidence Highlighting
   
## Step 8: Final Prediction
   - Save output as a file
    
## Step 9: Analyze Results
   - Check which types of events your model gets wrong (politics, finance, disasters).
   - Identify patterns
   - Diagrams | charts 

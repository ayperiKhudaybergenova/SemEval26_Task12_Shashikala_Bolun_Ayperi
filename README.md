# Task-12

The Abductive Event Reasoning (AER) task is framed as a multiple-choice question answering problem, aiming to evaluate large language models' ability to identify the most plausible direct cause of a real-world event based on textual evidence
for https://sites.google.com/view/semeval2026-task12/

datset https://github.com/sooo66/semeval2026-task12-dataset

#  TODO 




## Current Status (Overview)

-by 31.10.25    reading releated papers 

https://arxiv.org/pdf/2305.16646

https://aclanthology.org/2025.acl-long.1269.pdf?utm_source=chatgpt.com

-by 07.11.25   Running the first code with what provided.

-by 14.11.25   First presentation.


## Prioritized TODO List with Known Issues and details.

1. Literature review
   
   
 
## Main Schdeule from Class :
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
     
## Step 4: Prepare Model Input
   - Create prompt
     
## Step 5: Train the Model
   - Using train split
     
## Step 6: Evaluate the Model
   - Get predited asnwers
   - Compare with correct answers (Full match → 1.0 | Partial match → 0.5 | Wrong → 0.0)
     
## Step 7: Addional Evaluation (Extra)
   - Multi-Step Reasoning
   - Evidence Highlighting
   
## Step 8: Final Prediction
   - Save output as a file
    
## Step 9: Analyze Results
   - Check which types of events your model gets wrong (politics, finance, disasters).
   - Identify patterns
   - Diagrams | charts 

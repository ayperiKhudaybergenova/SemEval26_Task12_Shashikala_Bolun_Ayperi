"""
DeBERTa-v3 Official Metric Evaluator
====================================

This script evaluates trained model checkpoints against the official competition metric.
Unlike the training loss (which operates on individual options), this script:
1. Reconstructs the question-level structure from binary predictions.
2. Calculates the 'Exact Match' and 'Partial Match' scores.
3. Ranks all saved checkpoints to identify the best performing model.

Usage:
    python eval_official.py
"""

import json
import os
import glob
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# ================= Configuration =================
# Directory containing 'checkpoint-xxx' folders or 'final_model'
OUTPUT_DIR = "./ckpts" 

# Validation Dataset Paths
VAL_QWEN = "ToTval.json"
VAL_QS = "dataset/val_data/questions.jsonl"
VAL_DOCS = "dataset/val_data/docs.json"

MAX_LEN = 512
BATCH_SIZE = 32  # Larger batch size for faster inference

# ================= Data Utilities =================

def load_docs_map(docs_path):
    """
    Creates a mapping from topic_id to a concatenated string of document titles.
    """
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs_data = json.load(f)
    topic_map = {}
    for item in docs_data:
        t_id = item['topic_id']
        titles = [d.get('title', '') for d in item.get('docs', [])]
        titles_str = " | ".join([t for t in titles if t])
        topic_map[t_id] = titles_str
    return topic_map

def load_questions_list(qs_path):
    """
    Loads raw questions to retrieve Golden Answers for evaluation.
    """
    qs_list = []
    with open(qs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qs_list.append(json.loads(line))
    return qs_list

class MultiLabelVerifierDataset(Dataset):
    """
    Dataset class for Inference. 
    Constructs inputs identical to the training phase but includes metadata
    to allow regrouping predictions back to the Question level.
    """
    def __init__(self, qwen_path, qs_path, docs_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        topic_map = load_docs_map(docs_path)
        qs_list = load_questions_list(qs_path)
        
        with open(qwen_path, 'r', encoding='utf-8') as f:
            qwen_data = json.load(f)
            
        for item in qwen_data:
            q_idx = item.get("Question Index")
            if q_idx is None or q_idx >= len(qs_list): continue
            
            original_q = qs_list[q_idx]
            topic_id = original_q.get("topic_id")
            target_event = original_q.get("target_event", "")
            doc_titles = topic_map.get(topic_id, "")
            
            # Iterate through options A, B, C, D
            for opt_key in ["A", "B", "C", "D"]:
                opt_text = original_q.get(f"option_{opt_key}", "")
                if opt_key not in item: continue
                reasoning = item[opt_key].get("reasoning", "")
                
                # Input construction (Must match training logic!)
                input_text = (
                    f"Event: {target_event} [SEP] "
                    f"Option: {opt_text} [SEP] "
                    f"Logic: {reasoning} [SEP] "
                    f"Ref_Titles: {doc_titles}"
                )
                
                self.samples.append({
                    "text": input_text,
                    "label": 0, # Dummy label for inference
                    "meta": {   # Metadata for scoring
                        "q_idx": q_idx,
                        "opt": opt_key
                    }
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # No labels needed for prediction mode
        }

# ================= Scoring Logic (Official Metric) =================

def calculate_score(predictions, dataset, qs_list):
    """
    Aggregates binary predictions back to question-level sets and computes
    the Official System Score (Exact/Partial Match).
    """
    # 1. Group predictions by Question Index
    # Structure: { q_idx: { "pred_set": set(), "gold_set": set() } }
    q_results = {}
    
    # Convert Logits -> Class IDs (0 or 1)
    preds_binary = np.argmax(predictions.predictions, axis=1)
    
    for idx, pred_label in enumerate(preds_binary):
        meta = dataset.samples[idx]["meta"]
        q_idx = meta["q_idx"]
        opt = meta["opt"]
        
        if q_idx not in q_results:
            # Initialize Gold Set for this question
            original_q = qs_list[q_idx]
            golden_str = original_q.get("golden_answer", "")
            # Parse "A,B" -> {"A", "B"}
            gold_set = set(x.strip() for x in golden_str.replace("，", ",").split(",") if x.strip())
            q_results[q_idx] = {
                "pred_set": set(),
                "gold_set": gold_set
            }
        
        # If model predicts "True" (1), add this option to prediction set
        if pred_label == 1:
            q_results[q_idx]["pred_set"].add(opt)
            
    # 2. Calculate Score per Question
    scores = []
    for q_idx, data in q_results.items():
        p = data["pred_set"]
        g = data["gold_set"]
        
        # Rule 1: Exact Match (Full Points)
        if p == g:
            scores.append(1.0)
        # Rule 2: Partial Match (Subset & Non-empty)
        elif p.issubset(g) and len(p) > 0:
            scores.append(0.5)
        # Rule 3: Incorrect (Contains wrong option or Empty)
        else:
            scores.append(0.0)
            
    # Return average score
    return sum(scores) / len(scores) if scores else 0.0

# ================= Main Execution =================

def main():
    # 1. Locate all checkpoints
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    # Sort numerically (e.g., checkpoint-100, checkpoint-200)
    checkpoints.sort(key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0]))
    
    # Include 'final_model' if it exists
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    if os.path.exists(final_model_path):
        checkpoints.append(final_model_path)
        
    if not checkpoints:
        print(f"❌ No checkpoints found in {OUTPUT_DIR}")
        return

    print(f"🔎 Found {len(checkpoints)} models to evaluate.")
    
    # 2. Prepare Dataset (Load once to save time)
    print("📦 Loading Validation Dataset...")
    # Load tokenizer from the first checkpoint (vocab usually doesn't change)
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0]) 
    val_dataset = MultiLabelVerifierDataset(
        VAL_QWEN, VAL_QS, VAL_DOCS, tokenizer, max_len=MAX_LEN
    )
    qs_list = load_questions_list(VAL_QS) # For Gold Answers
    
    results_log = []

    # 3. Evaluate Loop
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\n🚀 Evaluating: {ckpt_name} ...")
        
        try:
            # Load Model
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
            model.eval() 
            
            # Run Inference
            trainer = Trainer(model=model)
            preds = trainer.predict(val_dataset)
            
            # Compute Metric
            score = calculate_score(preds, val_dataset, qs_list)
            print(f"   🏆 Official Score: {score:.4f}")
            
            results_log.append((ckpt_name, score))
            
            # Cleanup VRAM (Crucial for iterating many checkpoints)
            del model
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ⚠️ Error evaluating {ckpt_name}: {e}")

    # 4. Print Leaderboard
    print("\n" + "="*40)
    print("📊 FINAL LEADERBOARD (Official Metric)")
    print("="*40)
    
    # Sort by score descending
    results_log.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (name, score) in enumerate(results_log, 1):
        print(f"{rank}. {name} \t| Score: {score:.4f}")
    
    print("="*40)
    if results_log:
        best_model = results_log[0][0]
        print(f"✅ The BEST checkpoint is: {best_model}")

if __name__ == "__main__":
    main()
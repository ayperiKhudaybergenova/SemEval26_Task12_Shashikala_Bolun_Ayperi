import os
import sys

_local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

import json
import re
import torch
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from functools import partial

# ================= UNSLOTH INTEGRATION =================
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# Filter noisy warnings
warnings.filterwarnings("ignore", message=r".*An output with one or more elements was resized.*")
warnings.filterwarnings("ignore", message=r".*The installed version of bitsandbytes.*")

# ================= CONFIGURATION =================
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
TRAIN_DOCS_PATH = "datasets/train_data_ranked/docs.json"
TRAIN_QUES_PATH = "datasets/train_data_ranked/questions.jsonl"
OUTPUT_DIR = "checkpoints/qwen3-4b-aer-rl-unsloth-test"

# 🔥 Increased context length to 32K to accommodate long reference docs
MAX_SEQ_LENGTH = 16384  # Qwen3 supports up to 32K natively; 64K+ possible with YaRN (not needed yet)
MAX_COMPLETION_LENGTH = 8192  # JSON output is short; 512 is more than enough

# Optimization
LORA_RANK = 64
LEARNING_RATE = 5e-7
MAX_STEPS = 600
SAVE_STEPS = 20

# Batching (Unsloth + A100 80G)
PER_DEVICE_BATCH_SIZE = 4   # Reduced from 16 for better stability in RL
GRAD_ACCUM_STEPS = 2       # Keep effective batch size = 16
NUM_GENERATIONS = 4

SYSTEM_PROMPT = """You are an expert in Abductive Event Reasoning.
Your task is to analyze the provided context documents and identify **all and only** the direct causes of the target event among the given options (A, B, C, D).

Important rules:
- There may be **one or more correct answers**.
- You **MUST NOT select any incorrect option** — even one wrong choice results in zero reward.
- You **MUST select every correct option** — omitting a correct one reduces reward unless you avoid all errors.
- Base your judgment **strictly on evidence in the context documents**. Do not use external knowledge.

Output ONLY a valid JSON object with keys A, B, C, D.
Each value must be: {"reasoning": "concise justification (under 100 words)", "is_correct": true/false}.

!!!IMPORTANT: Keep your internal thinking and reasoning under 2048 tokens. The entire response must fit within 3072 tokens!!!
!!!Be concise. Long-winded thinking and reasoning will be penalized!!!
DO NOT include any text outside the JSON. DO NOT use markdown."""
print(f"[DEBUG] OUTPUT_DIR exists: {os.path.exists(OUTPUT_DIR)}")
print(f"[DEBUG] Checkpoint-210 exists: {os.path.exists(os.path.join(OUTPUT_DIR, 'checkpoint-210'))}")
print(f"[DEBUG] OUTPUT_DIR = {repr(OUTPUT_DIR)}")

# ================= DATA LOADING =================
def load_local_dataset():
    print(f"Loading data from {TRAIN_DOCS_PATH}...")
    with open(TRAIN_DOCS_PATH, 'r', encoding='utf-8') as f:
        docs_raw = json.load(f)
    topic_docs_map = {item['topic_id']: item['docs'] for item in docs_raw}
    
    data_samples = []
    with open(TRAIN_QUES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            tid = q.get('topic_id')
            docs = topic_docs_map.get(tid, [])
            data_samples.append({
                "target_event": q['target_event'],
                "docs": docs,
                "option_A": q.get('option_A', ''),
                "option_B": q.get('option_B', ''),
                "option_C": q.get('option_C', ''),
                "option_D": q.get('option_D', ''),
                "answer": q.get('golden_answer', "")
            })
    print(f"Loaded {len(data_samples)} training samples.")
    return Dataset.from_list(data_samples)

'''
def prompt_formatting_func(examples, tokenizer, max_seq_length, max_completion_length):
    """
    Build prompts with STRUCTURE-AWARE TOKEN-LEVEL TRUNCATION.
    - Preserves system/user/assistant structure.
    - Adds only COMPLETE documents (no mid-doc cuts).
    - Uses BGE-reranked doc order (high relevance first).
    """
    inputs = []
    max_prompt_tokens = max_seq_length - max_completion_length  # e.g., 32768 - 512 = 32256
    
    for i in range(len(examples['target_event'])):
        # Fixed prompt parts (MUST be preserved)
        sys_part = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        user_tail = (
            f"<|im_start|>user\n"
            f"Target Event: {examples['target_event'][i]}\n"
            f"Options:\n"
            f"A: {examples['option_A'][i]}\n"
            f"B: {examples['option_B'][i]}\n"
            f"C: {examples['option_C'][i]}\n"
            f"D: {examples['option_D'][i]}\n"
            "Analyze the context and output the JSON result.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # Token count for fixed parts
        fixed_text = sys_part + user_tail
        fixed_tokens = len(tokenizer.encode(fixed_text, add_special_tokens=False))
        available_for_docs = max_prompt_tokens - fixed_tokens - 10  # safety buffer
        
        if available_for_docs <= 0:
            raise ValueError("Fixed prompt too long! Reduce SYSTEM_PROMPT or increase MAX_SEQ_LENGTH.")
        
        # Add reranked docs one-by-one (only if fully fits)
        context_parts = []
        current_tokens = 0
        for doc in examples['docs'][i]:
            d_content = doc.get('content', '')[:5000]  # cap per-doc content
            d_text = f"Doc {doc.get('id', 'N/A')}: {doc.get('title', '')}\n{d_content}\n"
            d_tokens = len(tokenizer.encode(d_text, add_special_tokens=False))
            
            if current_tokens + d_tokens <= available_for_docs:
                context_parts.append(d_text)
                current_tokens += d_tokens
            else:
                break  # skip remaining (lower-relevance) docs
        
        full_prompt = sys_part + "".join(context_parts) + user_tail
        inputs.append(full_prompt)
    
    return {"prompt": inputs, "answer": examples['answer']}
'''
def prompt_formatting_func(examples, tokenizer, max_seq_length, max_completion_length):
    """
    Build prompts with STRUCTURE-AWARE TOKEN-LEVEL TRUNCATION.
    - Preserves system/user/assistant structure.
    - Adds only COMPLETE documents (no mid-doc cuts).
    - Uses BGE-reranked doc order (high relevance first).
    - Logs actual token usage for debugging.
    """
    inputs = []
    max_prompt_tokens = max_seq_length - max_completion_length  # e.g., 16384 - 2048 = 14336

    for i in range(len(examples['target_event'])):
        # Fixed prompt parts (MUST be preserved)
        sys_part = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        user_tail = (
            f"<|im_start|>user\n"
            f"Target Event: {examples['target_event'][i]}\n"
            f"Options:\n"
            f"A: {examples['option_A'][i]}\n"
            f"B: {examples['option_B'][i]}\n"
            f"C: {examples['option_C'][i]}\n"
            f"D: {examples['option_D'][i]}\n"
            "Analyze the context and output the JSON result.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # Token count for fixed parts
        fixed_text = sys_part + user_tail
        fixed_tokens = len(tokenizer.encode(fixed_text, add_special_tokens=False))
        available_for_docs = max_prompt_tokens - fixed_tokens - 10  # safety buffer
        
        if available_for_docs <= 0:
            raise ValueError("Fixed prompt too long! Reduce SYSTEM_PROMPT or increase MAX_SEQ_LENGTH.")
        
        # Add reranked docs one-by-one (only if fully fits)
        context_parts = []
        current_tokens = 0
        for doc in examples['docs'][i]:
            d_content = doc.get('content', '')[:5000]  # cap per-doc content to avoid outliers
            d_text = f"Doc {doc.get('id', 'N/A')}: {doc.get('title', '')}\n{d_content}\n"
            d_tokens = len(tokenizer.encode(d_text, add_special_tokens=False))
            
            if current_tokens + d_tokens <= available_for_docs:
                context_parts.append(d_text)
                current_tokens += d_tokens
            else:
                break  # skip remaining (lower-relevance) docs
        
        full_prompt = sys_part + "".join(context_parts) + user_tail
        inputs.append(full_prompt)
        
        # 🔍 DEBUG LOG: print actual token usage for first few samples
        if i < 10:  # log first 10 samples
            actual_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
            print(f"[DEBUG] Sample {i}: Prompt tokens = {actual_tokens} "
                  f"(max allowed: {max_prompt_tokens}) | Docs used: {len(context_parts)}")
    
    return {"prompt": inputs, "answer": examples['answer']}


# ================= REWARD FUNCTIONS =================

def extract_json_answer(completion):
    """
    Extracts the JSON object from the model's completion.
    Handles <think> tags and markdown code blocks.
    """
    # Remove thinking part if present to isolate the JSON output
    if "</think>" in completion:
        final_part = completion.split("</think>", 1)[-1]
    else:
        final_part = completion
        
    try:
        return json.loads(final_part)
    except:
        try:
            # Fallback: Try finding JSON inside markdown code blocks
            match = re.search(r'```json\s*(\{.*?\})\s*```', final_part, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Fallback: Try finding raw JSON structure
            match = re.search(r'\{.*\}', final_part, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
    return None

def compute_correctness_score(completion, gold_str):
    """
    Common helper function to evaluate the answer correctness.
    
    Args:
        completion (str): The model's raw output string.
        gold_str (str): The ground truth answer string (e.g., "A, B").
        
    Returns:
        tuple: (score [float], is_exact_match [bool])
            - score: 1.0 (Exact), 0.5 (Subset), 0.0 (Wrong/Empty)
            - is_exact_match: True only if the prediction exactly matches the gold set.
    """
    # Parse ground truth into a set
    gold_set = set([x.strip() for x in gold_str.split(',') if x.strip()])
    
    # Parse model prediction
    pred_json = extract_json_answer(completion)
    
    if not pred_json:
        return 0.0, False
        
    # Extract selected options where "is_correct": true
    pred_set = set()
    if isinstance(pred_json, dict):
        for opt in ['A', 'B', 'C', 'D']:
            item = pred_json.get(opt, {})
            # Check if the structure is valid and the option is marked correct
            if isinstance(item, dict) and item.get('is_correct') is True:
                pred_set.add(opt)
    
    # Evaluation Logic
    if not pred_set:
        return 0.0, False # No option selected
    elif pred_set == gold_set:
        return 1.0, True  # Exact Match
    elif pred_set.issubset(gold_set):
        return 0.5, False # Partial Match (Subset)
    else:
        return 0.0, False # Wrong (Includes incorrect options)

def semeval_score_reward(prompts, completions, answer, **kwargs):
    """
    Reward function for accuracy.
    - Exact match: 2.0
    - Subset match: 1.0
    - Wrong: 0.0
    """
    rewards = []
    for completion, gold_str in zip(completions, answer):
        # reuse helper: returns 1.0, 0.5, or 0.0
        score, _ = compute_correctness_score(completion, gold_str)
        rewards.append(score * 2.0) # Scale score by 2.0
    return rewards

def length_reward(prompts, completions, answer, tokenizer=None, **kwargs):
    """
    Proximity-Penalized Length Reward (Safe Version).
    Now accepts 'tokenizer' directly via partial binding.
    """
    rewards = []
    
    # Constants
    OPTIMAL_LENGTH = 2048
    ZERO_REWARD_THRESHOLD = 4096
    DANGER_THRESHOLD = 4096
    MAX_LIMIT = 8192
    MAX_PENALTY = 0.5

    # Debug print
    if not hasattr(length_reward, "has_printed"):
        if len(completions) > 0:
            sample_text = completions[0]
            if tokenizer:
                debug_count = len(tokenizer.encode(sample_text, add_special_tokens=False))
                debug_source = "Tokenizer (Accurate)"
            else:
                debug_count = len(sample_text) // 4
                debug_source = "Char//4 (Fallback)"

            print(f"\n[Reward-Debug] Source: {debug_source} | Count: {debug_count} | Threshold: {DANGER_THRESHOLD}")
            length_reward.has_printed = True

    for completion, gold_str in zip(completions, answer):
        # 1. Calculate Count
        if tokenizer:
            token_ids = tokenizer.encode(completion, add_special_tokens=False)
            token_count = len(token_ids)
        else:
            token_count = len(completion) // 4
        
        # 2. Check Correctness
        _, is_exact_match = compute_correctness_score(completion, gold_str)

        r = 0.0

        # === ZONE A: DANGER ZONE ===
        if token_count > DANGER_THRESHOLD:
            ratio = (token_count - DANGER_THRESHOLD) / (MAX_LIMIT - DANGER_THRESHOLD)
            ratio = min(ratio, 1.0) # Clamp at 1.0
            r = - (ratio * MAX_PENALTY) 
        
        # === ZONE B: BONUS ZONE ===
        elif is_exact_match:
            if token_count <= OPTIMAL_LENGTH:
                r = 0.5
            elif token_count <= ZERO_REWARD_THRESHOLD:
                r = 0.5 - (token_count - OPTIMAL_LENGTH) * (0.5 / (ZERO_REWARD_THRESHOLD - OPTIMAL_LENGTH))
            else:
                r = 0.0
        
        # === ZONE C: NEUTRAL ===
        else:
            r = 0.0
            
        rewards.append(r)
        
    return rewards

def format_reward(completions, **kwargs):
    """
    Simple format check: Reward 0.5 if JSON is parseable, else 0.0.
    """
    return [0.5 if extract_json_answer(c) is not None else 0.0 for c in completions]


# ================= MAIN =================
def main():
    print(f"🚀 Initializing Training with Unsloth + FlashAttention-2")
    print(f"Model: {MODEL_ID}")
    print(f"Context Window: {MAX_SEQ_LENGTH}")
    print(f"Max Completion Length: {MAX_COMPLETION_LENGTH}")

    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #torch.cuda.set_device(local_rank)
    #print(f"Process {local_rank} is using GPU: {torch.cuda.current_device()}")

    # 1. Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
        gpu_memory_utilization=0.75,
        #device_map = {"": local_rank},
        device_map = {"": 0},
    )

    # 2. Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Load and preprocess dataset
    dataset = load_local_dataset()
    
    # Inject tokenizer into map function via closure
    def map_with_tokenizer(examples):
        return prompt_formatting_func(
            examples,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH
        )

    length_reward_func = partial(length_reward, tokenizer=tokenizer)
    length_reward_func.__name__ = "length_reward"
    
    dataset = dataset.map(map_with_tokenizer, batched=True)
    dataset = dataset.map(lambda x: {'len': len(x['prompt'])}).sort('len', reverse=True)

    # 4. Training args
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        resume_from_checkpoint=True,
        run_name="qwen3-4b-aer-rl-unsloth",
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=6,
        logging_steps=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        bf16=is_bfloat16_supported(),
        tf32=is_bfloat16_supported(),
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_SEQ_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        use_vllm=False,
        report_to="none",
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, semeval_score_reward, length_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    # 6. Train
    print("🏁 Starting GRPO Training (Unsloth Optimized)...")
    #trainer.train()
    #trainer.train(resume_from_checkpoint=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, "fristrun")
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from specific checkpoint: {checkpoint-492}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("⚠️ Checkpoint not found, starting from scratch!")
        trainer.train()

    # 7. Save
    print("💾 Saving final adapter...")
    model.save_lora(os.path.join(OUTPUT_DIR, "final_checkpoint"))


if __name__ == "__main__":
    main()

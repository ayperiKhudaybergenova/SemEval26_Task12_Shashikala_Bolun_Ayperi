import os
import json
import torch
from tqdm import tqdm
from typing import List, Dict, Any

# --- Unsloth & Transformers Imports ---
from unsloth import FastLanguageModel
from transformers import TextStreamer

# ================= CONFIGURATION =================
# Paths (using absolute paths or relative to project root)
PROJECT_ROOT = os.path.expanduser("~/AER")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints/qwen3-4b-aer-rl-unsloth/checkpoint-492")
VAL_DOCS_PATH = os.path.join(PROJECT_ROOT, "datasets/val_data_ranked/docs.json")
VAL_QUES_PATH = os.path.join(PROJECT_ROOT, "datasets/val_data_ranked/questions.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "val_results_492.jsonl")

# Model Parameters
MAX_SEQ_LENGTH = 16384     # Must match training context window
DTYPE = None               # None = auto detection (Float16 or Bfloat16 based on GPU)
LOAD_IN_4BIT = False       # Set to False for maximum accuracy on A100 (BF16 is fast enough)

# Inference Parameters
BATCH_SIZE = 16            # A100 80G can easily handle 16-32 for this model size
MAX_NEW_TOKENS = 8192      # Enough for the JSON thinking process + answer
TEMPERATURE = 0.0          # 0.0 = Greedy decoding (deterministic, best for validation)
# If you want to evaluate RL exploration capability, set do_sample=True and temp=0.8

# System Prompt (MUST BE IDENTICAL TO TRAINING)
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

# ================= HELPER FUNCTIONS =================

def load_validation_data(docs_path: str, ques_path: str) -> List[Dict[str, Any]]:
    """
    Loads validation questions and merges them with their corresponding documents.
    """
    print(f"📂 Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Docs file not found: {docs_path}")
        
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs_raw = json.load(f)
    
    # Create a lookup map: topic_id -> list of docs
    topic_docs_map = {item['topic_id']: item['docs'] for item in docs_raw}
    
    samples = []
    print(f"📂 Loading questions from {ques_path}...")
    if not os.path.exists(ques_path):
        raise FileNotFoundError(f"Questions file not found: {ques_path}")

    with open(ques_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            q = json.loads(line)
            tid = q.get('topic_id')
            
            # Retrieve docs for this topic
            docs = topic_docs_map.get(tid, [])
            
            samples.append({
                "id": q.get('id'),
                "topic_id": tid,
                "target_event": q.get('target_event'),
                "options": {
                    "A": q.get('option_A', ''),
                    "B": q.get('option_B', ''),
                    "C": q.get('option_C', ''),
                    "D": q.get('option_D', '')
                },
                "docs": docs,
                "golden_answer": q.get('golden_answer', "")
            })
            
    print(f"✅ Loaded {len(samples)} validation samples.")
    return samples

def format_prompt(sample: Dict[str, Any], tokenizer) -> str:
    """
    Constructs the prompt exactly as done during training.
    """
    # 1. Fixed parts construction
    sys_part = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    user_tail = (
        f"<|im_start|>user\n"
        f"Target Event: {sample['target_event']}\n"
        f"Options:\n"
        f"A: {sample['options']['A']}\n"
        f"B: {sample['options']['B']}\n"
        f"C: {sample['options']['C']}\n"
        f"D: {sample['options']['D']}\n"
        "Analyze the context and output the JSON result.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 2. Calculate remaining budget for documents
    # Note: We use a safe buffer to ensure we don't exceed model limits
    max_prompt_tokens = MAX_SEQ_LENGTH - MAX_NEW_TOKENS
    fixed_text = sys_part + user_tail
    fixed_tokens = len(tokenizer.encode(fixed_text, add_special_tokens=False))
    available_for_docs = max_prompt_tokens - fixed_tokens - 100 # Safety buffer

    # 3. Add documents (Training logic: BGE-ranked order, truncate if full)
    context_parts = []
    current_tokens = 0
    
    for doc in sample['docs']:
        # Cap per-doc content similar to training script
        d_content = doc.get('content', '')[:5000] 
        d_text = f"Doc {doc.get('id', 'N/A')}: {doc.get('title', '')}\n{d_content}\n"
        d_tokens = len(tokenizer.encode(d_text, add_special_tokens=False))
        
        if current_tokens + d_tokens <= available_for_docs:
            context_parts.append(d_text)
            current_tokens += d_tokens
        else:
            # Stop adding docs if we run out of space (structure-aware truncation)
            break
            
    full_prompt = sys_part + "".join(context_parts) + user_tail
    return full_prompt

# ================= MAIN EXECUTION =================

def main():
    print("🚀 Initializing Inference...")
    
    # 1. Load Model & Tokenizer
    # using Unsloth's FastLanguageModel for optimization
    print(f"🔄 Loading checkpoint from: {CHECKPOINT_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CHECKPOINT_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # CRITICAL: Enable native inference optimization (2x speedup)
    FastLanguageModel.for_inference(model)
    print("⚡️ Unsloth Inference Optimizations Enabled.")

    # 2. Prepare Data
    raw_samples = load_validation_data(VAL_DOCS_PATH, VAL_QUES_PATH)
    
    # Pre-format all prompts to save time during the loop
    print("📝 Formatting prompts...")
    formatted_prompts = []
    for s in raw_samples:
        s['full_prompt'] = format_prompt(s, tokenizer)
        formatted_prompts.append(s)

    # 3. Inference Loop
    print(f"▶️ Starting inference on {len(formatted_prompts)} samples with Batch Size {BATCH_SIZE}...")
    
    # Clear output file if it exists (or choose to append if resuming)
    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️ Warning: Output file {OUTPUT_FILE} exists. Appending to it.")
    
    total_samples = len(formatted_prompts)
    
    # Iterate in batches
    for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Inferencing"):
        batch_samples = formatted_prompts[i : i + BATCH_SIZE]
        batch_prompts = [s['full_prompt'] for s in batch_samples]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        ).to("cuda")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                do_sample=False,       # Greedy decoding for validation
                temperature=0.0,       # Deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # Decode Output
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Process and Save Batch
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for j, raw_text in enumerate(decoded_outputs):
                sample_meta = batch_samples[j]
                
                # Robust extraction of the assistant's response
                # We split by the assistant tag to ignore the input prompt in the output
                if "<|im_start|>assistant" in raw_text:
                    response_text = raw_text.split("<|im_start|>assistant")[-1].strip()
                else:
                    # Fallback if tags are stripped unexpectedly
                    response_text = raw_text
                
                # Remove end tag if present
                response_text = response_text.replace("<|im_end|>", "").strip()

                result_entry = {
                    "id": sample_meta['id'],
                    "topic_id": sample_meta['topic_id'],
                    "golden_answer": sample_meta['golden_answer'],
                    "model_prediction_raw": response_text
                }
                
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Inference Complete!")
    print(f"💾 Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
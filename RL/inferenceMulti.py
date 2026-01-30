import os
import sys
import json
import time
from typing import List, Dict, Any
import torch.multiprocessing as mp

# ================= CONFIGURATION LIST =================
# Define your multiple datasets here, Workers will automatically load balance and process them
CONFIGS = [
    {
        "dataset_name": "train_set",
        "checkpoint_path": os.path.expanduser("~/AER/ckptBackup/4B/checkpoint-492"),
        "val_docs_path": os.path.expanduser("~/AER/datasets/train_data_ranked/docs.json"),
        "val_ques_path": os.path.expanduser("~/AER/datasets/train_data_ranked/questions.jsonl"),
        "output_file": os.path.expanduser("~/AER/train_results_492_qwen.jsonl"),
        "template_type": "qwen3",
        "max_seq_length": 16384,
        "max_new_tokens": 8192,
        "batch_size": 8,
        "temperature": 0.0,
        "load_in_4bit": False,
    },
    {
        "dataset_name": "test_set",
        "checkpoint_path": os.path.expanduser("~/AER/ckptBackup/4B/checkpoint-492"),
        "val_docs_path": os.path.expanduser("~/AER/datasets/test_data_ranked/docs.json"),
        "val_ques_path": os.path.expanduser("~/AER/datasets/test_data_ranked/questions.jsonl"),
        "output_file": os.path.expanduser("~/AER/test_results_492_qwen.jsonl"),
        "template_type": "qwen3",
        "max_seq_length": 16384,
        "max_new_tokens": 8192,
        "batch_size": 8,
        "temperature": 0.0,
        "load_in_4bit": False,
    }
]

# List of GPU IDs participating in competition
AVAILABLE_GPUS = ["0", "1"]

SYSTEM_PROMPT = """You are an expert in Abductive Event Reasoning.
Your task is to analyze the provided context documents and identify **all and only** the direct causes of the target event among the given options (A, B, C, D).

Important rules:
- There may be **one or more correct answers**.
- You **MUST NOT select any incorrect option** â€" even one wrong choice results in zero reward.
- You **MUST select every correct option** â€" omitting a correct one reduces reward unless you avoid all errors.
- Base your judgment **strictly on evidence in the context documents**. Do not use external knowledge.

Output ONLY a valid JSON object with keys A, B, C, D.
Each value must be: {"reasoning": "concise justification (under 100 words)", "is_correct": true/false}.

!!!IMPORTANT: Keep your internal thinking and reasoning under 2048 tokens. The entire response must fit within 3072 tokens!!!
!!!Be concise. Long-winded thinking and reasoning will be penalized!!!
DO NOT include any text outside the JSON. DO NOT use markdown."""

# ================= WORKER PROCESS (STRICTLY MODIFIED) =================

def inference_worker(gpu_id: str, task_queue, lock_manager, model_path: str, max_seq_len: int, load_4bit: bool):
    import os
    import sys
    import warnings

    # =============================================================================
    # 1. Pure Python preprocessing (zero dependencies)
    # =============================================================================
    # [Mamba Shield] 
    sys.modules["mamba_ssm"] = None 
    # --- 1. Physical environment isolation (must be before import torch) ---
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    import json
    from typing import List, Dict, Any
    
    # Delayed import in subprocess to ensure environment variables take effect
    import torch
    import torch.multiprocessing as mp
    from unsloth import FastLanguageModel
    from tqdm import tqdm

    # Now device is always cuda:0, because the process can only see one card
    device = "cuda:0"
    torch.cuda.empty_cache()
    
    # =============================================================================
    # 2. Model Initialization
    # =============================================================================
    print(f"âš™ï¸ [GPU {gpu_id}] Initializing model: {model_path}")
    
    # Must load model here to ensure it's in the correct CUDA context
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_len,
        load_in_4bit=load_4bit,
    )
    FastLanguageModel.for_inference(model)

    # =============================================================================
    # 3. Task Consumption Loop (Load Balancing)
    # =============================================================================
    while True:
        try:
            # Task package format: {"cfg": dataset_cfg, "samples": [batch_samples]}
            # Set timeout to prevent deadlock
            task_package = task_queue.get(timeout=20)
        except:
            # Queue is empty or timeout, exit
            break

        if task_package is None: # Received Poison Pill stop signal
            break

        cfg = task_package["cfg"]
        batch_samples = task_package["samples"]

        # --- Dynamic Prompt Formatting ---
        batch_prompts = []
        for s in batch_samples:
            options_text = (
                f"Target Event: {s['target_event']}\n"
                f"Options:\nA: {s['options']['A']}\nB: {s['options']['B']}\nC: {s['options']['C']}\nD: {s['options']['D']}\n"
                "Analyze the context and output the JSON result.\n"
            )
            
            if cfg['template_type'] == "qwen3":
                sys_p = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                tail = f"<|im_start|>user\n{options_text}<|im_end|>\n<|im_start|>assistant\n"
            else: # mistral
                sys_p = f"<s>[INST] {SYSTEM_PROMPT}\n\n"
                tail = f"{options_text} [/INST]"

            # Calculate budget
            max_p_tokens = cfg['max_seq_length'] - cfg['max_new_tokens'] - 200
            fixed_tokens = len(tokenizer.encode(sys_p + tail, add_special_tokens=False))
            available = max_p_tokens - fixed_tokens

            context_parts = []
            curr = 0
            for doc in s['docs']:
                d_text = f"Doc {doc.get('id', 'N/A')}: {doc.get('title', '')}\n{doc.get('content', '')[:5000]}\n"
                d_tokens = len(tokenizer.encode(d_text, add_special_tokens=False))
                if curr + d_tokens <= available:
                    context_parts.append(d_text)
                    curr += d_tokens
                else: break
            batch_prompts.append(sys_p + "".join(context_parts) + tail)

        # --- Inference ---
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg['max_seq_length']).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg['max_new_tokens'],
                use_cache=True,
                do_sample=False,
                temperature=cfg['temperature'],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # --- Safe File Writing ---
        # Get the corresponding lock based on dataset name, enabling parallel writes to different files, serial writes to same file
        lock = lock_manager[cfg['dataset_name']]
        with lock:
            with open(cfg['output_file'], 'a', encoding='utf-8') as f:
                for j, raw_text in enumerate(decoded):
                    if cfg['template_type'] == "qwen3":
                        resp = raw_text.split("assistant")[-1].strip()
                    else:
                        resp = raw_text.split("[/INST]")[-1].strip()
                    
                    # Clean possible end markers
                    resp = resp.replace("<|im_end|>", "").strip()
                    
                    f.write(json.dumps({
                        "id": batch_samples[j]['id'],
                        "golden_answer": batch_samples[j]['golden_answer'],
                        "model_prediction_raw": resp,
                        "processed_by_gpu": gpu_id
                    }, ensure_ascii=False) + "\n")

    print(f"ðŸ [GPU {gpu_id}] Worker finished all assigned tasks.")

# ================= MAIN CONTROLLER =================

def main():
    # Must set spawn in main process
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    
    # Task queue
    task_queue = manager.Queue()
    
    # Lock manager: allocate a separate lock for each dataset_name
    lock_manager = {cfg['dataset_name']: manager.Lock() for cfg in CONFIGS}

    print("ðŸ"¦ Pre-processing datasets and populating task pool...")
    
    total_batches = 0
    for cfg in CONFIGS:
        print(f"   -> Loading {cfg['dataset_name']} from {cfg['val_docs_path']}...")
        if not os.path.exists(cfg['val_docs_path']):
            raise FileNotFoundError(f"Missing docs: {cfg['val_docs_path']}")
            
        with open(cfg['val_docs_path'], 'r', encoding='utf-8') as f:
            docs_map = {item['topic_id']: item['docs'] for item in json.load(f)}
        
        samples = []
        with open(cfg['val_ques_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                q = json.loads(line)
                samples.append({
                    "id": q.get('id'),
                    "topic_id": q.get('topic_id'),
                    "target_event": q.get('target_event'),
                    "options": {"A": q.get('option_A'), "B": q.get('option_B'), "C": q.get('option_C'), "D": q.get('option_D')},
                    "docs": docs_map.get(q.get('topic_id'), []),
                    "golden_answer": q.get('golden_answer', "")
                })
        
        # Split into batches and enqueue
        # Each batch carries its own config (cfg), so Workers know what parameters to use
        curr_batches = 0
        for i in range(0, len(samples), cfg['batch_size']):
            task_queue.put({
                "cfg": cfg, 
                "samples": samples[i : i + cfg['batch_size']]
            })
            curr_batches += 1
        total_batches += curr_batches
        print(f"      Added {len(samples)} samples ({curr_batches} batches).")

    # Put poison pills (stop signals), one for each GPU
    for _ in AVAILABLE_GPUS:
        task_queue.put(None)

    print(f"ðŸš€ Launching {len(AVAILABLE_GPUS)} workers for {total_batches} total batches...")
    
    # Assume all tasks use the same base model checkpoint (prerequisite for load balancing)
    primary_model = CONFIGS[0]['checkpoint_path'] 
    
    processes = []
    for gpu_id in AVAILABLE_GPUS:
        p = mp.Process(target=inference_worker, args=(
            gpu_id, 
            task_queue, 
            lock_manager, 
            primary_model, 
            CONFIGS[0]['max_seq_length'], 
            CONFIGS[0]['load_in_4bit']
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("âœ¨ All datasets processed. Load balancing complete.")

if __name__ == "__main__":
    main()

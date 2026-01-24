import os
import sys
import warnings

# =============================================================================
# 1. Pure Python preprocessing (zero dependencies)
# =============================================================================
# [Mamba Shield] 
sys.modules["mamba_ssm"] = None 

# [Hard Masking] 
local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
print(f"🔒 [Rank {local_rank}] Hard Masking: Visible GPU = {local_rank}")

# [Critical parameter configuration]
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["TORCH_COMPILE_DISABLE_CACHE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.6" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# 2. Import Unsloth (MUST be first!)
# =============================================================================
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

# =============================================================================
# 3. Other dependencies
# =============================================================================
import torch
import json
import re
from accelerate import Accelerator
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from functools import partial

# Filter noisy warnings
warnings.filterwarnings("ignore", message=r".*An output with one or more elements was resized.*")
warnings.filterwarnings("ignore", message=r".*The installed version of bitsandbytes.*")
warnings.filterwarnings("ignore")

accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"🚀 [Master] Unsloth imported FIRST. Optimizations active.")

# =============================================================================
# 4. CONFIGURATION (migrated from trainRL.py)
# =============================================================================
MODEL_ID = "Qwen/Qwen3-32B"
TRAIN_DOCS_PATH = "datasets/train_data_ranked/docs.json"
TRAIN_QUES_PATH = "datasets/train_data_ranked/questions.jsonl"
OUTPUT_DIR = "checkpoints/qwen3-32B-aer-GRPO-NF4"

# 🔥 Increased context length to accommodate long reference docs
MAX_SEQ_LENGTH = 16384  # Qwen3 supports up to 32K natively
MAX_COMPLETION_LENGTH = 8192  # JSON output with reasoning

# Optimization
LORA_RANK = 64
LEARNING_RATE = 5e-7
MAX_STEPS = 500
SAVE_STEPS = 20

# Batching (adapted for vLLM multi-GPU)
PER_DEVICE_BATCH_SIZE = 2   # Conservative setting for vLLM
GRAD_ACCUM_STEPS = 4        # Maintain effective batch size
NUM_GENERATIONS = 4

SYSTEM_PROMPT = """You are an expert in Abductive Event Reasoning.
Your task is to analyze the provided context documents and identify **all and only** the direct causes of the target event among the given options (A, B, C, D).

Important rules:
- There may be **one or more correct answers**.
- You **MUST NOT select any incorrect option** — even one wrong choice results in zero reward.
- You **MUST select every correct option** — omitting a correct one reduces reward unless you avoid all errors.
- Base your judgment **strictly on evidence in the context documents**. Do not use external knowledge.

Output ONLY a valid JSON object with keys A, B, C, D.
Each value must be: {"reasoning": "concise justification (50 tokens for each reasoning)", "is_correct": true/false}.

!!!IMPORTANT: Be concise, Keep your internal thinking under 768 tokens, 512 is optimal, the more you exceed, the greater the penalty!!!
DO NOT include any text outside the JSON. DO NOT use markdown."""

# =============================================================================
# 5. DATA LOADING (migrated from trainRL.py)
# =============================================================================
def load_local_dataset():
    if local_rank == 0:
        print(f"Loading data from {TRAIN_DOCS_PATH}...")
    
    try:
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
        
        if local_rank == 0:
            print(f"Loaded {len(data_samples)} training samples.")
        
        return Dataset.from_list(data_samples)
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return Dataset.from_list([])

# =============================================================================
# 6. PROMPT FORMATTING (migrated from trainRL.py)
# =============================================================================
def prompt_formatting_func(examples, tokenizer, max_seq_length, max_completion_length):
    """
    Build prompts with STRUCTURE-AWARE TOKEN-LEVEL TRUNCATION.
    - Preserves system/user/assistant structure.
    - Adds only COMPLETE documents (no mid-doc cuts).
    - Uses BGE-reranked doc order (high relevance first).
    - Logs actual token usage for debugging.
    """
    inputs = []
    max_prompt_tokens = max_seq_length - max_completion_length  # e.g., 16384 - 8192 = 8192

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
            "<think>\n"
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
        if i < 10 and local_rank == 0:  # log first 10 samples on main process
            actual_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
            print(f"[DEBUG] Sample {i}: Prompt tokens = {actual_tokens} "
                  f"(max allowed: {max_prompt_tokens}) | Docs used: {len(context_parts)}")
    
    return {"prompt": inputs, "answer": examples['answer']}

# =============================================================================
# 7. REWARD FUNCTIONS (migrated from trainRL.py)
# =============================================================================
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
    Advanced Structural Length Reward.
    Splits evaluation into two parts:
    1. Thinking Process (<think>...): Encourages deep reasoning (Target ~2048 tokens).
    2. JSON Output ('reasoning' fields): Encourages detailed justification (Target ~400 tokens total).
    
    Range: Each part contributes between -0.4 and +0.4.
    Total Reward Range: [-0.8, +0.8].
    """
    rewards = []
    
    # --- Configuration for Thinking Process ---
    THINK_OPTIMAL = 512
    THINK_ZERO_THRESHOLD = 768
    THINK_DANGER_THRESHOLD = 768
    THINK_MAX_LIMIT = 1536
    THINK_MAX_REWARD = 0.5
    THINK_MAX_PENALTY = 0.5

    # --- Configuration for JSON Reasoning Fields ---
    JSON_OPTIMAL = 200
    JSON_LOWER_BOUND = 100
    JSON_UPPER_BOUND = 300
    JSON_MAX_REWARD = 0.3
    JSON_MIN_REWARD = -0.3 # Penalty floor

    # Debug helper to print only once
    if not hasattr(length_reward, "has_printed") and local_rank == 0:
        print(f"\n[Reward-Config] Thinking Target: {THINK_OPTIMAL} | JSON Reasoning Target: {JSON_OPTIMAL}")
        length_reward.has_printed = True

    for completion in completions:
        # =========================================================
        # PART 1: Extract and Evaluate Thinking Process
        # =========================================================
        think_content = ""
        # Robust extraction of <think> content
        if "</think>" in completion:
            raw_think = completion.split("</think>", 1)[0]
            think_content = raw_think.replace("<think>", "").strip()
        
        # Calculate Thinking Token Count
        if tokenizer:
            think_tokens = len(tokenizer.encode(think_content, add_special_tokens=False))
        else:
            think_tokens = len(think_content) // 4  # Fallback
            
        # Calculate Thinking Reward
        r_think = 0.0
        
        # Zone A: Bonus Zone (0 to THINK_ZERO_THRESHOLD)
        if think_tokens <= THINK_ZERO_THRESHOLD:
            if think_tokens <= THINK_OPTIMAL:
                # Perfect range: Full reward
                r_think = THINK_MAX_REWARD
            else:
                # Linear decay from Optimal to Zero
                decay_ratio = (think_tokens - THINK_OPTIMAL) / (THINK_ZERO_THRESHOLD - THINK_OPTIMAL)
                r_think = THINK_MAX_REWARD * (1.0 - decay_ratio)
        
        # Zone B: Penalty Zone (> THINK_DANGER_THRESHOLD)
        else:
            # Linear penalty from Zero to Max Limit
            penalty_ratio = (think_tokens - THINK_DANGER_THRESHOLD) / (THINK_MAX_LIMIT - THINK_DANGER_THRESHOLD)
            penalty_ratio = min(penalty_ratio, 1.0) # Clamp
            r_think = -1 * (penalty_ratio * THINK_MAX_PENALTY)

        # =========================================================
        # PART 2: Extract and Evaluate JSON Reasoning Fields
        # =========================================================
        json_reasoning_text = ""
        
        # Reuse existing extraction logic to find the JSON part
        json_obj = extract_json_answer(completion)
        
        if json_obj and isinstance(json_obj, dict):
            # Concatenate reasoning fields from A, B, C, D
            for key in ['A', 'B', 'C', 'D']:
                item = json_obj.get(key, {})
                if isinstance(item, dict):
                    json_reasoning_text += str(item.get("reasoning", "")) + " "
        
        # Calculate JSON Reasoning Token Count
        if tokenizer:
            json_tokens = len(tokenizer.encode(json_reasoning_text, add_special_tokens=False))
        else:
            json_tokens = len(json_reasoning_text) // 4

        # Calculate JSON Reward (Tent Function: Peak at 400)
        r_json = 0.0
        
        if json_tokens < JSON_LOWER_BOUND:
            # Too short (< 200): Flat penalty
            r_json = JSON_MIN_REWARD
        elif json_tokens > JSON_UPPER_BOUND:
            # Too long (> 600): Flat penalty
            r_json = JSON_MIN_REWARD
        else:
            # Inside the Tent [200, 600]
            distance = abs(json_tokens - JSON_OPTIMAL)
            max_dist = (JSON_UPPER_BOUND - JSON_LOWER_BOUND) / 2
            penalty_slope = (JSON_MAX_REWARD - JSON_MIN_REWARD) / max_dist
            r_json = JSON_MAX_REWARD - (distance * penalty_slope)

        # =========================================================
        # Combine Rewards
        # =========================================================
        total_r = r_think + r_json
        rewards.append(total_r)
        
        # Debug logging (occasional)
        if hasattr(length_reward, "has_printed") and length_reward.has_printed and kwargs.get("is_first_batch", False) and local_rank == 0:
             print(f"[Reward-Trace] Think: {think_tokens}tok -> {r_think:.3f} | JSON: {json_tokens}tok -> {r_json:.3f} | Total: {total_r:.3f}")

    return rewards

def format_reward(completions, **kwargs):
    """
    Simple format check: Reward 0.5 if JSON is parseable, else 0.0.
    """
    return [0.5 if extract_json_answer(c) is not None else 0.0 for c in completions]

# =============================================================================
# 8. DDP HEALTH CHECK CALLBACK
# =============================================================================
from transformers import TrainerCallback
import torch.distributed as dist

class DDPHealthCheckCallback(TrainerCallback):
    """
    Deep DDP Health Check:
    1. Verify communication link survival (Heartbeat).
    2. Verify if LoRA weights across GPUs are identical (Weight Sync Check).
    """
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Frequency control: Check every 5 steps (too frequent slows down training; too sparse detects issues late)
        #if state.global_step % 5 != 0:
        #    return
        try:
            # --- 1. Preparation ---
            # Get current process rank
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            
            # Skip check if in single-GPU mode
            if world_size < 2:
                return

            # --- 2. Extract Model Fingerprint ---
            # CRITICAL: Must find a parameter with requires_grad=True (i.e., LoRA layers).
            # Frozen base parameters never change; checking them won't detect gradient sync failures.
            check_param = None
            param_name = "Unknown"
            
            # Attempt to unwrap potential model wrappers (DDP, PeftModel, etc.) used by Unsloth
            base_model = model.module if hasattr(model, "module") else model
            
            for name, p in base_model.named_parameters():
                if p.requires_grad:
                    check_param = p
                    param_name = name
                    break
            
            if check_param is None:
                # Fallback: If no trainable params found (rare), take the first parameter
                check_param = next(base_model.parameters())
                if rank == 0:
                    print(f"⚠️ [DDP Warning] No trainable params found? Checking frozen param instead.")

            # Use the first value of the parameter tensor as the fingerprint
            local_val = check_param.data.view(-1)[0].item()
            
            # --- 3. Construct Check Tensor ---
            # [0]: Local fingerprint value, [1]: Counter 1.0 (to calculate total rank count)
            # Must be on cuda:0 (due to Hard Masking)
            check_tensor = torch.tensor([local_val, 1.0], device="cuda:0", dtype=torch.float32)

            # --- 4. Execute Collective Communication (All-Reduce) ---
            # This step blocks until all GPUs submit data. If it hangs here, the link is broken.
            dist.all_reduce(check_tensor, op=dist.ReduceOp.SUM)

            # --- 5. Main Process Verification ---
            if rank == 0:
                global_sum_val = check_tensor[0].item()
                global_count = check_tensor[1].item()
                
                # Theoretical sum (if no Split Brain, values across cards should be identical)
                expected_sum = local_val * world_size
                
                # A. Verify node count (Prevent Split Brain)
                if int(global_count) != world_size:
                    print(f"\n❌ [DDP FATAL] Split Brain detected! Expected {world_size} GPUs, found {int(global_count)}.")
                    # You could raise an exception here to abort training
                
                # B. Verify weight consistency (Verify gradient sync)
                # Allow tiny floating-point errors (BF16 precision is approx 1e-2 ~ 1e-3)
                elif abs(global_sum_val - expected_sum) > 1e-3:
                    print(f"\n❌ [DDP FATAL] Weight Divergence! Gradients are NOT syncing.")
                    print(f"   - Local (Rank 0): {local_val:.6f}")
                    print(f"   - Global Avg:     {global_sum_val / world_size:.6f}")
                    print(f"   - Param checked:  {param_name}")
                
                # C. All clear
                else:
                    # Print only at lower logging levels to avoid cluttering
                    pass 
                    print(f"✅ [DDP OK] Step {state.global_step}: Sync verified on {param_name}")

        except Exception as e:
            print(f"\n❌ [DDP ERROR] Health check failed: {str(e)}")

# =============================================================================
# 9. MAIN TRAINING FUNCTION
# =============================================================================
def main():
    """
    Main training function that orchestrates the entire GRPO training pipeline.
    Steps:
    1. Load model with Unsloth optimizations
    2. Prepare dataset with proper synchronization
    3. Configure GRPO trainer
    4. Execute training with checkpoint resume support
    5. Save final model
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Model Loading
    # -------------------------------------------------------------------------
    if local_rank == 0:
        print(f"🔥 Loading Model: {MODEL_ID}")
        print(f"Context Window: {MAX_SEQ_LENGTH}")
        print(f"Max Completion Length: {MAX_COMPLETION_LENGTH}")

    current_device = "cuda:0"  # Hard Masking takes effect, unified to cuda:0

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.35, 
        device_map={"": current_device},
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # -------------------------------------------------------------------------
    # Step 2: Dataset Preparation
    # -------------------------------------------------------------------------
    # 🔥 [Data Synchronization] Wrap complex processing logic with main_process_first
    with accelerator.main_process_first():
        dataset = load_local_dataset()

        if len(dataset) > 0:
            # Inject tokenizer into map function via closure (fully reuse your code)
            def map_with_tokenizer(examples):
                return prompt_formatting_func(
                    examples,
                    tokenizer=tokenizer,
                    max_seq_length=MAX_SEQ_LENGTH,
                    max_completion_length=MAX_COMPLETION_LENGTH
                )
            
            # Prepare length_reward with tokenizer (fully reuse your code)
            length_reward_func = partial(length_reward, tokenizer=tokenizer)
            length_reward_func.__name__ = "length_reward"
            
            # Execute Map and Sort (ordering is critical for batch packing efficiency)
            dataset = dataset.map(map_with_tokenizer, batched=True, load_from_cache_file=True)
            dataset = dataset.map(lambda x: {'len': len(x['prompt'])}).sort('len', reverse=True)
            
            if local_rank == 0:
                print(f"✅ Dataset prepared: {len(dataset)} samples (Sorted by length)")
        else:
            # Fallback to prevent empty dataset from causing reference errors below
            length_reward_func = partial(length_reward, tokenizer=tokenizer)

    # -------------------------------------------------------------------------
    # Step 3: Training Configuration
    # -------------------------------------------------------------------------
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        resume_from_checkpoint=True,  # Enable auto-resume
        run_name="qwen3-32b-aer-GRPO-NF4",
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=25,
        logging_steps=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        bf16=is_bfloat16_supported(),
        tf32=is_bfloat16_supported(),
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_SEQ_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        # vLLM configuration
        use_vllm=True,
        vllm_gpu_memory_utilization=0.6, 
        report_to="none",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    # -------------------------------------------------------------------------
    # Step 4: Trainer Initialization
    # -------------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, semeval_score_reward, length_reward_func],
        args=training_args,
        train_dataset=dataset,
        callbacks=[DDPHealthCheckCallback()],
    )

    # -------------------------------------------------------------------------
    # Step 5: Training Execution with Checkpoint Resume
    # -------------------------------------------------------------------------
    if accelerator.is_main_process:
        print("🎯 Starting GRPO Training (Unsloth + vLLM Optimized)...")

    # Checkpoint resume mechanism
    checkpoint_path = os.path.join(OUTPUT_DIR, "none")  # Modify to your desired checkpoint

    if os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"📄 Resuming from specific checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        if local_rank == 0:
            print("⚠️ Checkpoint not found, starting from scratch!")
        trainer.train()

    # -------------------------------------------------------------------------
    # Step 6: Save Final Model
    # -------------------------------------------------------------------------
    if accelerator.is_main_process:
        print("💾 Saving final adapter...")
        model.save_lora(os.path.join(OUTPUT_DIR, "final_checkpoint"))
        print("✅ Training completed!")


# =============================================================================
# 10. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

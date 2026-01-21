import json
import os
import re

# ================= CONFIGURATION =================
# Adjust your paths here
INPUT_FILE = os.path.expanduser("~/AER/val_results_492.jsonl")
OUTPUT_SCORE_FILE = os.path.expanduser("~/AER/val_scores_492.json")

def parse_gold(gold_input):
    """
    Standardizes the golden answer to a set of uppercase letters.
    """
    if isinstance(gold_input, list):
        return set(gold_input)
    # Filter only A-D characters
    clean = re.sub(r'[^A-D]', '', str(gold_input).upper())
    return set(clean)

def extract_answer_content(raw_text):
    """
    Extracts the final JSON answer by stripping out the 'Thinking Process'.
    Logic aligns with trainRL.py: Split by the end-of-thought token.
    """
    content = raw_text
    
    # 1. Split by </think> tag if present
    # The model output structure is: [System/User Prompt] <think>...reasoning...</think> [JSON Answer]
    if "</think>" in content:
        content = content.split("</think>")[-1]
    
    # 2. Strip standard markdown code blocks if present
    # Often models wrap the final output in ```json ... ```
    content = content.strip()
    if content.startswith("```"):
        # Remove the first line (e.g., ```json)
        content = content.split("\n", 1)[-1]
        # Remove the last line (e.g., ```) if it exists
        if content.endswith("```"):
            content = content.rsplit("\n", 1)[0]
            
    return content.strip()

def parse_prediction(raw_json_str):
    """
    Parses the cleaned string into a set of predicted options.
    """
    # 1. Clean the text (remove thinking process)
    clean_text = extract_answer_content(raw_json_str)
    
    predicted_set = set()
    try:
        # 2. Load JSON
        data = json.loads(clean_text)
        
        # 3. Extract keys A, B, C, D where is_correct is True
        for key in ['A', 'B', 'C', 'D']:
            if key in data:
                item = data[key]
                is_correct = item.get('is_correct', False)
                
                # Handle boolean or string "true"
                if str(is_correct).lower() in ['true', '1', 'yes']:
                    predicted_set.add(key)
                    
    except json.JSONDecodeError:
        # If parsing fails, it means the model didn't output valid JSON after </think>
        return None
    except Exception:
        return None
        
    return predicted_set

def calculate_score(pred_set, gold_set):
    """
    SemEval Scoring Rules:
    1.0: Exact Match (P == G)
    0.5: Proper Subset (P < G)
    0.0: Incorrect (Any wrong option selected, or empty/parse error)
    """
    if pred_set is None: return 0.0
    if not pred_set: return 0.0
    
    # If prediction has ANY element not in Gold -> Score 0
    if not pred_set.issubset(gold_set):
        return 0.0
    
    # If prediction is exactly Gold -> Score 1.0
    if pred_set == gold_set:
        return 1.0
    
    # If prediction is a subset (and not empty, and no wrong answers) -> Score 0.5
    return 0.5

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File not found: {INPUT_FILE}")
        return

    print(f"🚀 Starting Evaluation (Logic: Split by </think>)")
    print(f"📂 Reading: {INPUT_FILE}")
    
    total_score = 0.0
    count = 0
    parse_errors = 0
    
    # Stats counters
    stats = {1.0: 0, 0.5: 0, 0.0: 0}
    results_detail = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            try:
                item = json.loads(line)
                
                gold_set = parse_gold(item.get('golden_answer', ''))
                raw_pred = item.get('model_prediction_raw', '')
                
                # --- CORE LOGIC ---
                pred_set = parse_prediction(raw_pred)
                # ------------------
                
                if pred_set is None:
                    parse_errors += 1
                    # Treat parse error as empty set for scoring (Score 0)
                    score = 0.0
                else:
                    score = calculate_score(pred_set, gold_set)
                
                total_score += score
                count += 1
                stats[score] = stats.get(score, 0) + 1
                
                results_detail.append({
                    "id": item.get("id"),
                    "gold": sorted(list(gold_set)),
                    "pred": sorted(list(pred_set)) if pred_set is not None else "PARSE_ERROR",
                    "score": score
                })
                
            except json.JSONDecodeError:
                continue

    if count == 0:
        print("⚠️ No valid data found.")
        return

    avg = total_score / count
    
    print("-" * 40)
    print(f"📊 REPORT (Samples: {count})")
    print("-" * 40)
    print(f"🏆 Average Score: {avg:.4f}")
    print("-" * 40)
    print(f"✅ Exact Matches (1.0):   {stats[1.0]} ({stats[1.0]/count:.1%})")
    print(f"⚠️ Partial Matches (0.5): {stats[0.5]} ({stats[0.5]/count:.1%})")
    print(f"❌ Zero Scores (0.0):     {stats[0.0]} ({stats[0.0]/count:.1%})")
    print(f"💀 JSON Parse Errors:     {parse_errors}")
    print("-" * 40)
    
    # Save details
    with open(OUTPUT_SCORE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"summary": {"avg": avg, "stats": str(stats)}, "details": results_detail}, f, indent=2)
        
    print(f"💾 Saved details to: {OUTPUT_SCORE_FILE}")

if __name__ == "__main__":
    main()
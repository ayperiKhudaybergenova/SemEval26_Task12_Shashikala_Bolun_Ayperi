import json
import requests
import re
import random
import sys
import logging
import os
from datetime import datetime
from typing import Dict, List, Any

# ================= CONFIGURATION =================
# Input and Output files
INPUT_FILE = 'docs.json'
OUTPUT_FILE = 'docs_final.json'
LOG_FILE = 'qc_filter.log'

# Ollama API Settings
OLLAMA_API_URL = "http://localhost:11434/api/chat"
# The exact model name you deployed
MODEL_NAME = "qwen3-30b-Instruct-Q4KM-128K" 

# Context Window Management
MAX_DOC_CONTENT_LENGTH = 2000 
# =================================================

# --- LOGGER SETUP ---
def setup_logger():
    """
    Configures a dual-output logger:
    1. FileHandler: Writes to LOG_FILE with timestamps.
    2. StreamHandler: Writes to Console (clean output).
    """
    logger = logging.getLogger('QC_Filter')
    logger.setLevel(logging.INFO)
    
    # Reset handlers if re-running in same session
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. File Handler (Detailed with Timestamp)
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # 2. Console Handler (Clean, like print)
    console_handler = logging.StreamHandler(sys.stdout)
    console_fmt = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def query_ollama(topic_name: str, docs: List[Dict]) -> Dict[str, str]:
    """
    Sends a topic and its documents to Ollama for classification.
    Returns a dictionary mapping UUID -> Label.
    """
    docs_text = ""
    for d in docs:
        content_preview = d.get('content', '')[:MAX_DOC_CONTENT_LENGTH].replace('\n', ' ')
        docs_text += f"--- Document ID: {d['uuid']} ---\nTitle: {d.get('title', 'N/A')}\nContent: {content_preview}\n\n"

    system_prompt = (
        "You are a strict Data Quality Control Agent.\n"
        "Classify each document regarding the Target Topic:\n"
        "- 'gold': DIRECTLY discusses the main event.\n"
        "- 'hard': SAME Entity but DIFFERENT event/time.\n"
        "- 'soft': COMPETITOR or RELATED domain.\n"
        "- 'irrelevant': Unrelated, spam, or broken text.\n"
        "\n"
        "Respond with JSON: {\"uuid1\": \"gold\", \"uuid2\": \"hard\", ...}"
    )

    user_prompt = (
        f"Target Topic: \"{topic_name}\"\n\n"
        f"Analyze these {len(docs)} documents:\n\n"
        f"{docs_text}"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "temperature": 0.1, 
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        content = result.get('message', {}).get('content', '')
        return parse_llm_json(content)
    except Exception as e:
        logger.error(f"  [Error] Ollama call failed: {e}")
        return {}

def parse_llm_json(content: str) -> Dict[str, str]:
    try:
        clean_content = re.sub(r'```json\s*|\s*```', '', content).strip()
        start = clean_content.find('{')
        end = clean_content.rfind('}') + 1
        if start != -1 and end != -1:
            return json.loads(clean_content[start:end])
        return json.loads(clean_content)
    except json.JSONDecodeError:
        return {}

def normalize_label(label: str) -> str:
    label = label.lower()
    if 'gold' in label: return 'gold'
    if 'hard' in label: return 'hard'
    if 'soft' in label: return 'soft'
    return 'irrelevant'

def main():
    logger.info(f"--- Starting QC Process at {datetime.now()} ---")
    logger.info(f"Loading {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error("Input file not found.")
        return

    logger.info(f"Connecting to Ollama ({MODEL_NAME})...")
    logger.info(f"Processing {len(dataset)} topics...\n")

    topic_stats = []

    # --- PHASE 1: LLM CLASSIFICATION ---
    for i, topic_entry in enumerate(dataset):
        topic_name = topic_entry.get('topic')
        docs = topic_entry.get('docs', [])
        
        logger.info(f"[{i+1}/{len(dataset)}] QC processing: {topic_name}")
        
        if not docs:
            logger.warning("  -> No docs found, skipping.")
            continue

        classification_map = query_ollama(topic_name, docs)
        
        counts = {'gold': 0, 'hard': 0, 'soft': 0, 'irrelevant': 0}
        
        for doc in docs:
            uuid_str = doc.get('uuid')
            raw_label = classification_map.get(uuid_str, 'irrelevant')
            label = normalize_label(raw_label)
            
            doc['qc_label'] = label # Tag the doc
            counts[label] += 1
        
        logger.info(f"  -> Labels: G:{counts['gold']} | H:{counts['hard']} | S:{counts['soft']} | I:{counts['irrelevant']}")
        
        topic_stats.append({
            'topic_data': topic_entry,
            'counts': counts
        })

    # --- PHASE 2: REPORTING ---
    report_header = "\n" + "="*80 + "\n"
    report_header += f"{'ID':<5} | {'Topic Name':<50} | {'Gold':<4} | {'Hard':<4} | {'Soft':<4} | {'Irr':<4}\n"
    report_header += "-" * 80
    logger.info(report_header)
    
    for stat in topic_stats:
        t_id = stat['topic_data'].get('topic_id')
        name = stat['topic_data'].get('topic')[:47] + "..."
        c = stat['counts']
        logger.info(f"{t_id:<5} | {name:<50} | {c['gold']:<4} | {c['hard']:<4} | {c['soft']:<4} | {c['irrelevant']:<4}")
    
    logger.info("="*80 + "\n")

    # --- PHASE 3: USER INPUT (THRESHOLDS & CAPS) ---
    logger.info("--- FILTERING CONFIGURATION ---")
    logger.info("(User input required in console)")
    
    try:
        # 1. Quality Gate
        min_gold_threshold = int(input("1. [Quality Gate] Minimum 'Gold' docs required to KEEP a topic (e.g., 5): "))
        logger.info(f"User selected Min Gold Threshold: {min_gold_threshold}")
        
        # 2. Quantity Caps (Random Sampling)
        print("\n--- SAMPLING LIMITS (Set max docs per category) ---")
        max_gold = int(input("2. Max 'Gold' docs to keep (e.g., 15): "))
        max_hard = int(input("3. Max 'Hard' docs to keep (e.g., 10): "))
        max_soft = int(input("4. Max 'Soft' docs to keep (e.g., 5): "))
        max_irr  = int(input("5. Max 'Irrelevant' docs to keep (usually 0): "))
        
        logger.info(f"User selected Caps -> Gold:{max_gold}, Hard:{max_hard}, Soft:{max_soft}, Irr:{max_irr}")

    except ValueError:
        logger.error("Invalid input. Please enter integers only. Exiting.")
        return

    final_dataset = []
    dropped_topics = 0

    logger.info("\nApplying filters and random sampling...")

    for stat in topic_stats:
        topic_entry = stat['topic_data']
        counts = stat['counts']
        
        # 1. Filter Topic
        if counts['gold'] < min_gold_threshold:
            dropped_topics += 1
            continue

        # 2. Random Sampling
        doc_buckets = {'gold': [], 'hard': [], 'soft': [], 'irrelevant': []}
        for doc in topic_entry['docs']:
            label = doc.get('qc_label', 'irrelevant')
            doc_buckets[label].append(doc)
        
        final_docs = []
        
        def sample_docs(docs_list, max_limit):
            if len(docs_list) > max_limit:
                random.shuffle(docs_list) 
                return docs_list[:max_limit]
            else:
                return docs_list

        final_docs.extend(sample_docs(doc_buckets['gold'], max_gold))
        final_docs.extend(sample_docs(doc_buckets['hard'], max_hard))
        final_docs.extend(sample_docs(doc_buckets['soft'], max_soft))
        final_docs.extend(sample_docs(doc_buckets['irrelevant'], max_irr))
        
        random.shuffle(final_docs)
        
        topic_entry['docs'] = final_docs
        final_dataset.append(topic_entry)

    # --- PHASE 4: SAVE ---
    summary = "\n" + "="*30 + "\n"
    summary += f"Original Topics: {len(topic_stats)}\n"
    summary += f"Final Topics:    {len(final_dataset)}\n"
    summary += f"Dropped Topics:  {dropped_topics}\n"
    summary += "="*30
    logger.info(summary)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)
        
    logger.info(f"\n[Success] Filtered dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
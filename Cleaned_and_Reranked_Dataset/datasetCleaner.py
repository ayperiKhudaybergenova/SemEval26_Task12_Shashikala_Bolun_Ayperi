import json
import os
import requests
import re
from tqdm import tqdm
from typing import List, Dict, Any

# ================= CONFIGURATION =================
# Ollama Settings
OLLAMA_API_URL = "http://localhost:11434/api/chat"
# Ensure this name matches exactly what is shown in 'ollama list'
MODEL_NAME = "qwen3-30b-Instruct-Q4KM-128K"  

# Data Paths
DATA_PATHS = [
    {
        "name": "TRAIN",
        "docs_path": "datasets/train_data/docs.json",
        "questions_path": "datasets/train_data/questions.jsonl",
        "out_dir": "datasets/train_data_cleaned"
    },
    {
        "name": "VAL",
        "docs_path": "datasets/val_data/docs.json",
        "questions_path": "datasets/val_data/questions.jsonl",
        "out_dir": "datasets/val_data_cleaned"
    }
]

# LLM Generation Parameters
CONTEXT_WINDOW = 128000  # Qwen support
MAX_DOC_LENGTH = 10000    # Truncate individual docs to prevent overflow
# =================================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_json(path):
    """Load standard JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_json_from_text(text):
    """Extract JSON list from LLM response text."""
    try:
        # Attempt to find [ "doc-1", ... ] pattern
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        # Fallback: if regex fails, return empty list to trigger safety mechanism
        return []
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return []

def call_ollama_filter(topic_info: str, events: List[str], docs: List[Dict]) -> List[str]:
    """
    Call Ollama to filter documents based on causal relevance.
    """
    
    # Construct the document text block
    docs_text = ""
    doc_id_map = {} # Used to validate returned IDs
    
    for doc in docs:
        d_id = doc.get('id')
        d_title = doc.get('title', 'No Title')
        # Truncate to avoid context overflow from a single massive document
        d_content = doc.get('content', '')[:MAX_DOC_LENGTH] 
        
        doc_id_map[d_id] = True
        docs_text += f"--- Document ID: {d_id} ---\nTitle: {d_title}\nContent: {d_content}\n\n"

    # Construct the Prompts
    events_text = "\n".join([f"- {e}" for e in events])
    
    system_prompt = (
        "You are an expert data filter for an Abductive Event Reasoning task. "
        "Your job is to read a list of retrieved documents and select ONLY those that are relevant to the target events.\n"
        "Criteria for KEEPING a document:\n"
        "1. Direct Cause: Directly explains why the event happened.\n"
        "2. Indirect Cause: Competitor actions, market trends, or regulatory changes that could trigger the event (Keep these!).\n"
        "3. Context: Necessary background info to understand the entities involved.\n"
        "Criteria for DISCARDING:\n"
        "1. Totally irrelevant topics (e.g., a cooking recipe for a tech event).\n"
        "2. Duplicate information that provides no new context.\n"
        "\n"
        "Return the output as a JSON list of strings containing the IDs of the documents to keep. "
        "Example: [\"doc-001\", \"doc-003\"]"
    )
    
    user_prompt = (
        f"Topic Category: {topic_info}\n"
        f"Target Events:\n{events_text}\n\n"
        f"Here are the Retrieved Documents:\n{docs_text}\n\n"
        f"Which documents should be KEPT? Return JSON list only."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "temperature": 0.1, # Low temperature for deterministic output
        "format": "json"    # Enforce JSON output (Ollama feature)
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=600) # 10-minute timeout
        response.raise_for_status()
        result = response.json()
        print(f"DEBUG: Prompt Tokens used: {result.get('prompt_eval_count')}")
        content = result['message']['content']
        
        keep_ids = extract_json_from_text(content)
        
        # Validate that returned IDs actually exist in the input
        valid_ids = [uid for uid in keep_ids if uid in doc_id_map]
        
        # Fallback Strategy: If LLM returns nothing or fails to parse, 
        # keep the first 5 docs (assuming original order has some relevance).
        if not valid_ids:
            print("Warning: Model returned no valid IDs or parse failed. Keeping first 5 docs as fallback.")
            return [d['id'] for d in docs[:5]]
            
        return valid_ids

    except Exception as e:
        print(f"\nError calling Ollama: {e}")
        # On network error, keep ALL docs to prevent data loss
        return [d['id'] for d in docs]

def process_dataset(config):
    print(f"Processing {config['name']} Data...")
    print(f"Loading from: {config['docs_path']} & {config['questions_path']}")
    
    ensure_dir(config['out_dir'])
    
    # 1. Load Data
    try:
        docs_data = load_json(config['docs_path'])
    except Exception:
        # Try loading as JSONL if standard JSON fails
        docs_data = load_jsonl(config['docs_path'])
        
    questions_data = load_jsonl(config['questions_path'])
    
    # 2. Map Questions to Topics
    # We need to know which events belong to which topic to provide context for the LLM
    topic_to_events = {}
    for q in questions_data:
        tid = q.get('topic_id')
        event = q.get('target_event')
        if tid not in topic_to_events:
            topic_to_events[tid] = set()
        topic_to_events[tid].add(event)
    
    # 3. Clean Documents
    cleaned_docs_data = []
    
    # Use tqdm to show progress bar
    for topic_item in tqdm(docs_data, desc=f"Cleaning {config['name']} Docs"):
        topic_id = topic_item.get('topic_id')
        topic_desc = topic_item.get('topic', '')
        original_docs = topic_item.get('docs', [])
        
        # Skip if no documents
        if not original_docs:
            cleaned_docs_data.append(topic_item)
            continue
            
        # Get all events associated with this topic
        events = list(topic_to_events.get(topic_id, ["Unknown Event"]))
        
        # Filter Logic:
        # If there are 3 or fewer docs, keep them all (no need to filter).
        if len(original_docs) <= 3:
            keep_ids = [d['id'] for d in original_docs]
        else:
            keep_ids = call_ollama_filter(topic_desc, events, original_docs)
        
        # Create filtered list
        filtered_docs = [d for d in original_docs if d['id'] in keep_ids]
        
        # Build new topic object
        new_topic_item = topic_item.copy()
        new_topic_item['docs'] = filtered_docs
        cleaned_docs_data.append(new_topic_item)
        
    # 4. Save Results
    out_docs_file = os.path.join(config['out_dir'], "docs.json")
    out_questions_file = os.path.join(config['out_dir'], "questions.jsonl")
    
    print(f"Saving cleaned docs to {out_docs_file}")
    with open(out_docs_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_docs_data, f, indent=2, ensure_ascii=False)
        
    print(f"Copying questions to {out_questions_file}")
    with open(out_questions_file, 'w', encoding='utf-8') as f:
        for q in questions_data:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
            
    print(f"Done with {config['name']}.\n")

if __name__ == "__main__":
    # Check Ollama Connection
    try:
        requests.get(OLLAMA_API_URL.replace("/api/chat", ""))
        print("Ollama connection successful.")
    except Exception:
        print("Error: Could not connect to Ollama. Please run 'ollama serve'.")
        exit(1)

    for conf in DATA_PATHS:
        process_dataset(conf)

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ================= CONFIGURATION =================
# BGE-Reranker-v2-m3 is a state-of-the-art multilingual reranker
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Paths to your cleaned data
DATA_PATHS = [
    {
        "docs": "datasets/train_data_cleaned/docs.json",
        "questions": "datasets/train_data_cleaned/questions.jsonl",
        "out": "datasets/train_data_ranked/docs.json" # Output to 'final' directory
    },
    {
        "docs": "datasets/val_data_cleaned/docs.json",
        "questions": "datasets/val_data_cleaned/questions.jsonl",
        "out": "datasets/val_data_ranked/docs.json"
    }
]
# =================================================

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(docs_path, questions_path):
    # Load Docs
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs_data = json.load(f) # List of topics
    
    # Load Questions to map TopicID -> Event
    # We use the event description as the query to score the documents
    topic_queries = {}
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            tid = q.get('topic_id')
            event = q.get('target_event')
            # Using the first event of the topic as the query is sufficient
            if tid not in topic_queries:
                topic_queries[tid] = event
                
    return docs_data, topic_queries

def main():
    print(f"Loading Reranker: {MODEL_NAME} ...")
    # Using bfloat16 for acceleration (native support on A100)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,  # 'torch_dtype' is correct for standard HF versions
        device_map="cuda",           # Requires 'pip install accelerate'
        trust_remote_code=True
    )
    model.eval()

    for conf in DATA_PATHS:
        print(f"\nProcessing {conf['docs']} ...")
        docs_data, topic_queries = load_data(conf['docs'], conf['questions'])
        
        sorted_docs_data = []
        
        for topic_item in tqdm(docs_data, desc="Reranking"):
            tid = topic_item.get('topic_id')
            original_docs = topic_item.get('docs', [])
            
            if not original_docs:
                sorted_docs_data.append(topic_item)
                continue
                
            # 1. Construct Query
            # Priority: Event description > Topic title
            query = topic_queries.get(tid, topic_item.get('topic', ''))
            
            # 2. Construct Pairs [Query, Doc_Content]
            pairs = []
            for doc in original_docs:
                # We only take the first ~2000 chars to judge relevance.
                # BGE-M3 supports long context, but truncation improves speed significantly.
                content = f"{doc.get('title', '')} {doc.get('content', '')}"[:2000]
                pairs.append([query, content])
            
            # 3. Inference / Scoring
            # Manual batching to prevent OOM (some topics have many docs)
            scores = []
            batch_size = 16 
            
            with torch.no_grad():
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i : i+batch_size]
                    inputs = tokenizer(
                        batch_pairs, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt', 
                        max_length=1024
                    ).to('cuda')
                    
                    batch_scores = model(**inputs, return_dict=True).logits.view(-1,).float()
                    scores.extend(batch_scores.cpu().numpy().tolist())

            # 4. Sort
            # Zip docs and scores together
            doc_score_pairs = list(zip(original_docs, scores))
            # Sort by score in descending order (Highest relevance first)
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Unpack
            sorted_docs = [p[0] for p in doc_score_pairs]
            
            # Update Topic object
            new_item = topic_item.copy()
            new_item['docs'] = sorted_docs
            sorted_docs_data.append(new_item)
            
        # 5. Save
        ensure_dir(conf['out'])
        print(f"Saving sorted docs to {conf['out']}")
        with open(conf['out'], 'w', encoding='utf-8') as f:
            json.dump(sorted_docs_data, f, indent=2, ensure_ascii=False)
            
        # Copy questions file to keep directory structure consistent
        out_q_path = conf['out'].replace("docs.json", "questions.jsonl")
        os.system(f"cp {conf['questions']} {out_q_path}")

    print("\n✅ Reranking Complete. Documents are now sorted by relevance.")
    print("Strategy: You can now safely truncate context to 8192 tokens in RL.")

if __name__ == "__main__":
    main()
'''
autoEval_MoA_qwen.py

Evaluation script using a Multi-Agent / Mixture of Experts approach.
It delegates the reasoning process to 'collaborative_inference.py'.
'''

import json
import os
import logging
import re
import glob
from datetime import datetime
import collaborative_inference  # Import the new module

# --- Configuration Section ---

# Expert System Configuration
# The model used for the 4 experts (Precondition, Commonsense, Temporal, Discourse)
EXPERT_MODEL_NAME = "qwen3-30b-Instruct-Q4KM-128K"
# The model used to synthesize answers (can be the same or a larger model)
AGGREGATOR_MODEL_NAME = "qwen3-30b-Thinking-Q4KM-256K" 

# Concurrency Settings
# 1 = Sequential (Low VRAM usage, stable)
# 4 = Full Parallel (High VRAM usage, faster if hardware supports it)
MAX_CONCURRENT_EXPERTS = 1

# Data source for the main evaluation
EVAL_DATA_DIR = "datasets/val_data"
EVAL_QUESTION_FILE = os.path.join(EVAL_DATA_DIR, "questions.jsonl")
EVAL_DOCS_FILE = os.path.join(EVAL_DATA_DIR, "docs.json")

# --- Helper Functions ---

def find_and_parse_latest_log():
    """Finds the latest log file and parses its state based on question indices."""
    log_files = glob.glob('evaluation_log_*.log')
    if not log_files:
        return None, set(), 0.0, 0

    latest_log_file = sorted(log_files, reverse=True)[0]
    
    processed_indices = set()
    last_score = 0.0
    last_count = 0

    index_pattern = re.compile(r"INFO - Question \(Index: (\d+),")
    summary_pattern = re.compile(r"INFO - Evaluated (\d+) questions \| Current Score: ([\d.]+) \|")

    try:
        with open(latest_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if index_match := index_pattern.search(line):
                    processed_indices.add(int(index_match.group(1)))
                if summary_match := summary_pattern.search(line):
                    last_count = int(summary_match.group(1))
                    last_score = float(summary_match.group(2))
        return latest_log_file, processed_indices, last_score, last_count
    except Exception as e:
        print(f"Error parsing log file {latest_log_file}: {e}")
        return None, set(), 0.0, 0

def load_data(file_path):
    """Loads data from a .jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_full_json(file_path):
    """Loads data from a single standard .json file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_docs_by_topic_id(topic_id, all_docs):
    """Finds the corresponding documents for a given topic_id."""
    for doc_item in all_docs:
        if doc_item.get("topic_id") == topic_id:
            # Remove imageUrl to save tokens and avoid confusion
            return [{k: v for k, v in doc.items() if k != 'imageUrl'} for doc in doc_item.get("docs", [])]
    return []

def parse_model_response(response_text):
    """Parses the model's JSON output."""
    if not response_text:
        return None
        
    try:
        # Attempt to parse pure JSON
        json_response = json.loads(response_text)
        if isinstance(json_response, dict) and "selected_options" in json_response:
            return sorted([opt.upper() for opt in json_response["selected_options"]])
    except json.JSONDecodeError:
        # Attempt to find JSON block within text
        try:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}') + 1
            if start_index != -1 and end_index != -1:
                json_str = response_text[start_index:end_index]
                json_response = json.loads(json_str)
                if isinstance(json_response, dict) and "selected_options" in json_response:
                    return sorted([opt.upper() for opt in json_response["selected_options"]])
        except (json.JSONDecodeError, KeyError):
            logging.warning(f"Could not parse JSON from irregular output: {response_text}")
            return None
    logging.warning(f"Could not parse model output: {response_text}")
    return None

def calculate_score(predicted_answers, golden_answers_str):
    """Calculates the score for a single question."""
    if not predicted_answers: return 0.0
    golden_answers = sorted(golden_answers_str.split(','))
    predicted_set, golden_set = set(predicted_answers), set(golden_answers)
    if predicted_set == golden_set: return 1.0
    if predicted_set.intersection(golden_set): return 0.5
    return 0.0

# --- Main Evaluation Logic ---

def main():
    log_to_resume, processed_indices, initial_score, initial_count = find_and_parse_latest_log()

    log_filename = log_to_resume or f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename, mode='a', encoding='utf-8'), logging.StreamHandler()])

    if log_to_resume:
        logging.info("="*50 + f"\nResuming from latest log file: {log_to_resume}\nRestored state: {initial_count} questions processed, total score {initial_score:.2f}\nWill skip {len(processed_indices)} already evaluated questions.\n" + "="*50)
    else:
        logging.info("No logs to resume from, starting a new evaluation task...")

    logging.info(f"Starting CoT-SC Mixture-of-Agents Evaluation...\nExperts: {EXPERT_MODEL_NAME}\nAggregator: {AGGREGATOR_MODEL_NAME}\nConcurrency: {MAX_CONCURRENT_EXPERTS}")

    # Initialize the Collaborative Inference Engine
    engine = collaborative_inference.ExpertReasoningEngine(
        expert_model=EXPERT_MODEL_NAME,
        aggregator_model=AGGREGATOR_MODEL_NAME,
        concurrency=MAX_CONCURRENT_EXPERTS,
        timeout=300 # Generous timeout for experts
    )

    try:
        questions, all_docs = load_data(EVAL_QUESTION_FILE), load_full_json(EVAL_DOCS_FILE)
    except FileNotFoundError as e:
        logging.error(f"Error: Evaluation data files not found - {e}")
        return

    total_score, processed_questions_count = initial_score, initial_count

    for i, question in enumerate(questions):
        if i in processed_indices:
            continue

        topic_id = question["topic_id"]
        logging.info(f"--- Evaluating Question {processed_questions_count + 1}/{len(questions)} (Question Index: {i}) (Topic ID: {topic_id}) ---")

        context_docs = get_docs_by_topic_id(topic_id, all_docs)
        if not context_docs:
            logging.warning(f"No documents found for Topic ID {topic_id}, skipping question.")
            continue

        # --- Logic Change: Delegate to Engine ---
        model_output = None
        try:
            # The engine handles the 4 experts + 1 aggregator loop internally
            model_output = engine.solve(question, context_docs)
            logging.info("Collaborative inference completed successfully.")
        except Exception as e:
            logging.error(f"Critical error in inference engine for question {i}: {e}")
        
        # ----------------------------------------

        if model_output is None:
            processed_questions_count += 1
            logging.warning(f"Failed to get model response for question {i}. Score: 0.")
            continue

        try:
            logging.info(f"Aggregator raw output: {model_output}")
            predicted_answer = parse_model_response(model_output)

            # Retry logic if JSON parsing fails (simple retry with the aggregator only)
            if predicted_answer is None:
                logging.warning("Model did not return valid JSON, attempting formatting retry...")
                # We construct a simple retry prompt manually here just to save the JSON
                retry_prompt = "Your previous output format was incorrect. Please strictly adhere to the JSON format: {\"reasoning\": \"...\", \"selected_options\": [\"A\"]}"
                
                try:
                    retry_response = ollama.chat(model=AGGREGATOR_MODEL_NAME, messages=[
                        {'role': 'assistant', 'content': model_output},
                        {'role': 'user', 'content': retry_prompt}
                    ])
                    retry_content = retry_response['message']['content']
                    logging.info(f"Retry output: {retry_content}")
                    predicted_answer = parse_model_response(retry_content)
                except Exception as re:
                    logging.error(f"Retry failed: {re}")

            golden_answer = question["golden_answer"]
            score = calculate_score(predicted_answer, golden_answer)
            
            total_score += score
            processed_questions_count += 1
            
            logging.info(f"Question (Index: {i}, Topic ID: {topic_id}): Prediction: {predicted_answer}, Golden: {golden_answer.split(',')}, Score: {score}")

        except Exception as e:
            logging.error(f"An error occurred during post-processing for question (Index: {i}): {e}")
            processed_questions_count += 1

        if processed_questions_count > 0:
            current_accuracy = total_score / processed_questions_count
            logging.info(f"Evaluated {processed_questions_count} questions | Current Score: {total_score:.2f} | Current Accuracy: {current_accuracy:.2%}\n")

    logging.info("--- Evaluation Complete ---")
    if processed_questions_count > 0:
        final_accuracy = total_score / processed_questions_count
        logging.info(f"Final Score: {total_score:.2f}\nTotal Questions Processed: {processed_questions_count}\nFinal Accuracy: {final_accuracy:.2%}")

if __name__ == "__main__":
    main()

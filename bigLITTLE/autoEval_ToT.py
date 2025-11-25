import ollama
import json
import os
import logging
import re
import glob
from datetime import datetime

# --- Configuration Section ---

# Model and data settings
MODEL_NAME = "qwen3-30b-Thinking-Q4KM-256K"

# --- Data source for the main evaluation ---
EVAL_DATA_DIR = "datasets/val_data"
EVAL_QUESTION_FILE = os.path.join(EVAL_DATA_DIR, "questions.jsonl")
EVAL_DOCS_FILE = os.path.join(EVAL_DATA_DIR, "docs.json")

# --- Timeout and retry settings ---
# Timeout for the model request in seconds
REQUEST_TIMEOUT = 360

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
            return [{k: v for k, v in doc.items() if k != 'imageUrl'} for doc in doc_item.get("docs", [])]
    return []

def get_system_prompt():
    """
    Defines the system prompt implementing the Tree of Thoughts (ToT) methodology.
    Refactored to use the 'BECAUSE' Test and Inclusive Causal Logic to prevent over-pruning.
    """
    return """
You are an expert Logician specializing in **Comprehensive Causal Analysis**.
Your task is to evaluate four candidate statements to determine if they are valid explanations for a Target Event based *strictly* on the provided context documents.

--- KEY PRINCIPLE: THE "BECAUSE" TEST ---
For each option, ask: **"Did the Target Event happen BECAUSE of this option?"**
*   If the Option is a **Root Cause** (e.g., Extreme Cold) -> **TRUE**.
*   If the Option is the **Direct Mechanism** (e.g., Grid Failure) -> **TRUE**.
*   If the Option is a **Constituent Part** of the event description (e.g., "The coup involved arrests") -> **TRUE**.
*   If the Option is a **Consequence/Aftermath** (e.g., "People died *after* the storm") -> **FALSE**.

--- TREE OF THOUGHTS PROTOCOL ---
Execute the following 3 stages strictly:

**Stage 1: Fact Verification & Timeline Check**
- Verify if the statement in the Option is **Factually True** according to the documents.
- Check the **Timeline**: Did the Option occur *before* or *during* the Target Event? 
- If an option is factually false or happened *after* the event (as a result), mark it as **FALSE**.

**Stage 2: The "BECAUSE" Test (Causal Logic)**
- For factually true options, apply the "BECAUSE" sentence test:
  - "Target Event happened **BECAUSE** [Option]."
- **CRITICAL**: Do not look for the "single best" cause. Often, an event has multiple valid explanations (e.g., both the Weather AND the Power Cut are valid causes for a Blackout).
- **CRITICAL**: Distinguish **CAUSE** vs **CONSEQUENCE**. 
  - If the Option happened *as a result* of the Event, it is **FALSE** (unless the question asks for what constitutes the event).

**Stage 3: Final Decision**
- Mark ALL options that pass the "BECAUSE" test as **TRUE**.
- Mark options that fail the test or are mere consequences as **FALSE**.

--- OUTPUT FORMAT ---
Output a JSON object evaluating every option.
{
  "A": {
    "is_correct": boolean,
    "reasoning": "1. Timeline Check: [Before/During/After]. 2. Because Test: [Does 'Event because Option' make sense?]. Cite Doc IDs."
  },
  "B": { "is_correct": boolean, "reasoning": "..." },
  "C": { "is_correct": boolean, "reasoning": "..." },
  "D": { "is_correct": boolean, "reasoning": "..." }
}
"""

def construct_prompt(question_item, context_docs):
    """Builds the prompt with explicit instructions to look for multiple valid parts of the causal chain."""
    event = question_item["target_event"]
    option_a = question_item["option_A"]
    option_b = question_item["option_B"]
    option_c = question_item["option_C"]
    option_d = question_item["option_D"]

    formatted_docs = "\n".join([
        f"--- Document ID: {i+1} ---\n{json.dumps(doc, ensure_ascii=False, indent=2)}" 
        for i, doc in enumerate(context_docs)
    ])

    user_prompt = f"""
Analyze the relationship between the Candidate Options and the Target Event using the **"BECAUSE" Test**.

**Target Event (The Effect to be Explained):**
{event}

**Available Context Documents (Evidence):**
{formatted_docs}

**Candidate Options:**
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

**Critical Instructions:**
1. **Accept Multiple Layers of Causality**: A "Root Cause" (e.g., heavy rain) and a "Direct Trigger" (e.g., dam breaking) are **BOTH** correct explanations for a flood. Do not discard one just because the other exists.
2. **Reject Consequences**: If an option describes what happened *after* the event (the outcome), it is **FALSE**.
3. **Strict Fact Checking**: If the option contradicts the document details (dates, names, locations), it is **FALSE**.

Return the JSON evaluation.
"""
    return user_prompt

def parse_model_response(response_text):
    """
    Parses the detailed JSON output to extract selected options for scoring.
    Expected format: {"A": {"is_correct": false, ...}, "B": {"is_correct": true, ...}, ...}
    """
    selected_options = []
    try:
        # Attempt to clean up markdown code blocks if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        # Locate the JSON object
        start_index = cleaned_text.find('{')
        end_index = cleaned_text.rfind('}') + 1
        
        if start_index != -1 and end_index != -1:
            json_str = cleaned_text[start_index:end_index]
            json_data = json.loads(json_str)
            
            # Iterate through keys A, B, C, D to find correct answers
            for key in ["A", "B", "C", "D"]:
                if key in json_data:
                    item = json_data[key]
                    # Handle boolean true/false or string "true"/"false"
                    is_correct = item.get("is_correct", False)
                    if isinstance(is_correct, str):
                        is_correct = is_correct.lower() == "true"
                    
                    if is_correct:
                        selected_options.append(key)
            
            return sorted(selected_options)
            
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.warning(f"JSON Parsing Error: {e} | Text excerpt: {response_text[:100]}...")
        return None
    
    return sorted(selected_options) if selected_options else None # Return None if parsing failed logically, or empty list if no options selected (which implies wrong answer)

def calculate_score(predicted_answers, golden_answers_str):
    """Calculates the score for a single question."""
    # If predicted_answers is None (parsing failure), score is 0
    if predicted_answers is None: return 0.0
    
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

    logging.info(f"Starting model evaluation (ToT Mode - Detailed JSON Output)...\nModel: {MODEL_NAME}\nDataset: {EVAL_DATA_DIR}")

    try:
        questions, all_docs = load_data(EVAL_QUESTION_FILE), load_full_json(EVAL_DOCS_FILE)
    except FileNotFoundError as e:
        logging.error(f"Error: Evaluation data files not found - {e}")
        return

    client = ollama.Client(timeout=REQUEST_TIMEOUT)
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

        user_prompt = construct_prompt(question, context_docs)
        messages = [
            {'role': 'system', 'content': get_system_prompt()},
            {'role': 'user', 'content': user_prompt}
        ]
        
        model_output = None
        for attempt in range(2):
            try:
                logging.info(f"Calling model... (Attempt {attempt + 1}/2)")
                
                response = client.chat(model=MODEL_NAME, messages=messages, think=True)
                
                model_output = response['message']['content']
                
                # Capture reasoning for logging
                thinking_content = response['message'].get('thinking')
                
                if not thinking_content:
                    think_match = re.search(r'<think>(.*?)</think>', model_output, re.DOTALL)
                    if think_match:
                        thinking_content = think_match.group(1).strip()
                
                if thinking_content:
                    logging.info(f"Model Thinking Process (ToT Trace):\n{thinking_content}")
                else:
                    logging.info("No thinking process detected (model might have skipped it).")

                logging.info(f"Model returned successfully.")
                break 
            except Exception as e:
                logging.error(f"Error calling model on attempt {attempt + 1}/2: {e}")
                if attempt == 1:
                    logging.error(f"Both attempts failed, skipping question (Index: {i}).")

        if model_output is None:
            processed_questions_count += 1
            logging.warning(f"Failed to get model response for question {i}. Score: 0.")
            if processed_questions_count > 0:
                 current_accuracy = total_score / processed_questions_count
                 logging.info(f"Evaluated {processed_questions_count} questions | Current Score: {total_score:.2f} | Current Accuracy: {current_accuracy:.2%}\n")
            continue

        try:
            logging.info(f"Model raw output: {model_output}")
            predicted_answer = parse_model_response(model_output)

            if predicted_answer is None:
                logging.warning("Model did not return valid JSON, retrying with format instruction...")
                retry_prompt = """Your output format was incorrect. Please strictly adhere to the specific JSON format required.
Return a JSON object where keys are "A", "B", "C", "D".
Example:
{
  "A": {"is_correct": false, "reasoning": "..."},
  "B": {"is_correct": true, "reasoning": "..."},
  ...
}"""
                
                retry_messages = messages + [
                    {'role': 'assistant', 'content': model_output},
                    {'role': 'user', 'content': retry_prompt}
                ]
                
                response = client.chat(model=MODEL_NAME, messages=retry_messages, think=True)
                model_output = response['message']['content']
                
                thinking_content_retry = response['message'].get('thinking')
                if not thinking_content_retry:
                    think_match = re.search(r'<think>(.*?)</think>', model_output, re.DOTALL)
                    if think_match:
                        thinking_content_retry = think_match.group(1).strip()
                
                if thinking_content_retry:
                    logging.info(f"Model Thinking Process (Retry):\n{thinking_content_retry}")

                logging.info(f"Model retry output: {model_output}")
                predicted_answer = parse_model_response(model_output)

            # Check if prediction is strictly None again (failed parsing), default to empty list implies wrong answer but valid JSON
            final_pred = predicted_answer if predicted_answer is not None else []
            
            golden_answer = question["golden_answer"]
            score = calculate_score(final_pred, golden_answer)
            
            total_score += score
            processed_questions_count += 1
            
            logging.info(f"Question (Index: {i}, Topic ID: {topic_id}): Prediction: {final_pred}, Golden: {golden_answer.split(',')}, Score: {score}")

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
    else:
        logging.info("No questions were processed.")

if __name__ == "__main__":
    main()

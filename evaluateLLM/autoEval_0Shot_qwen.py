'''

Zero-shot evaluation for model running on ollama.

'''

import ollama
import json
import os
import logging
import re
import glob
from datetime import datetime

# --- Configuration Section ---

# Model and data settings
MODEL_NAME = "qwen3-30b-Q4KM-256K"

# --- Data source for the main evaluation ---
EVAL_DATA_DIR = "datasets/sample_data"
EVAL_QUESTION_FILE = os.path.join(EVAL_DATA_DIR, "questions.jsonl")
EVAL_DOCS_FILE = os.path.join(EVAL_DATA_DIR, "docs.json")

# --- Timeout and retry settings ---
# Timeout for the model request in seconds (e.g., 600 seconds = 10 minutes)
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

def construct_prompt(question_item, context_docs):
    """Builds the user prompt for a given question and its context."""
    event = question_item["target_event"]
    option_a = question_item["option_A"]
    option_b = question_item["option_B"]
    option_c = question_item["option_C"]
    option_d = question_item["option_D"]

    formatted_docs = "\n\n".join([f"--- Document ---\n{json.dumps(doc, ensure_ascii=False, indent=2)}" for doc in context_docs])

    user_prompt = f"""
Based on the provided context documents, please analyze the event and select one or more of the most plausible explanations from the four options.

**Event Description:**
{event}

**Context Documents:**
{formatted_docs}

**Candidate Options:**
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

Please reason carefully and return your answer ONLY in JSON format.
"""
    return user_prompt

def get_system_prompt():
    """Defines the system prompt."""
    return """
You are a meticulous AI analytical assistant. Your task is to select the most plausible explanation for a given event, based exclusively on the provided context information.

Read all documents carefully and use logical reasoning.
Your output MUST strictly follow the JSON format below, with no additional explanations or text outside the JSON structure.
{
  "reasoning": "State your brief reasoning process here.",
  "selected_options": ["A", "B"]
}
Where `selected_options` is a list containing the letters of your chosen options.
"""

def parse_model_response(response_text):
    """Parses the model's JSON output."""
    try:
        json_response = json.loads(response_text)
        if isinstance(json_response, dict) and "selected_options" in json_response:
            return sorted([opt.upper() for opt in json_response["selected_options"]])
    except json.JSONDecodeError:
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

    logging.info(f"Starting model evaluation...\nModel: {MODEL_NAME}\nDataset: {EVAL_DATA_DIR}")

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
        # Construct the messages list for a zero-shot call
        messages = [
            {'role': 'system', 'content': get_system_prompt()},
            {'role': 'user', 'content': user_prompt}
        ]
        
        model_output = None
        for attempt in range(2):
            try:
                logging.info(f"Calling model... (Attempt {attempt + 1}/2)")
                response = client.chat(model=MODEL_NAME, messages=messages)
                model_output = response['message']['content']
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
                retry_prompt = "Your output format was incorrect. Please strictly adhere to the JSON format and provide only the JSON object:\n" + \
                               '{"reasoning": "Your reasoning here.", "selected_options": ["A"]}'
                
                # We add the bad response and the retry prompt to the original `messages` list
                # This helps the model understand its mistake in context
                retry_messages = messages + [
                    {'role': 'assistant', 'content': model_output},
                    {'role': 'user', 'content': retry_prompt}
                ]
                response = client.chat(model=MODEL_NAME, messages=retry_messages)
                model_output = response['message']['content']
                logging.info(f"Model retry output: {model_output}")
                predicted_answer = parse_model_response(model_output)

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
    else:
        logging.info("No questions were processed.")

if __name__ == "__main__":
    main()

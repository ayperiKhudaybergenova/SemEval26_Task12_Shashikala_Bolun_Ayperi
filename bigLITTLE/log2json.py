import re
import json
import argparse
import sys
import ast

def extract_json_from_text(text):
    """
    Extracts and parses a JSON object from mixed text.
    Handles Markdown code blocks and non-standard formatting.
    """
    if not text:
        return {}

    cleaned_text = text.strip()
    # 1. Remove Markdown code block markers
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    
    # 2. Find the outermost curly braces {}
    start = cleaned_text.find('{')
    end = cleaned_text.rfind('}')
    
    if start != -1 and end != -1:
        json_str = cleaned_text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: Attempt to use ast.literal_eval for Python-style dicts (e.g., single quotes)
            try:
                return ast.literal_eval(json_str)
            except:
                return {}
    return {}

def calculate_official_score(prediction, golden):
    """
    Recalculates the score based on the Official Metric.
    
    Metric Rules:
    1.0 (Full Match): If Prediction (P) is exactly equal to Golden (G).
    0.5 (Partial Match): If P is a non-empty proper subset of G (P subset G and P != G).
                         This means P contains at least one correct option and NO incorrect options.
    0.0 (Incorrect): Otherwise (including empty predictions or predictions with incorrect options).
    """
    # Handle None or empty inputs
    if prediction is None:
        return 0.0
    
    # Ensure inputs are sets for comparison
    try:
        p_set = set(prediction)
        g_set = set(golden)
    except TypeError:
        # If input is not iterable (e.g., parsing failed), score is 0
        return 0.0
    
    # Filter only valid labels to be strict
    valid_labels = {'A', 'B', 'C', 'D'}
    p_set = {x for x in p_set if x in valid_labels}
    g_set = {x for x in g_set if x in valid_labels}

    # If prediction is empty -> Incorrect (Rule 3)
    if not p_set:
        return 0.0

    # Rule 1: Full Match
    if p_set == g_set:
        return 1.0
    
    # Rule 2: Partial Match (Non-empty proper subset)
    # issubset() checks if all elements of p_set are in g_set.
    # Since we already checked p_set == g_set above, if issubset returns True here, it is a proper subset.
    if p_set.issubset(g_set):
        return 0.5
        
    # Rule 3: Incorrect (Contains wrong options)
    return 0.0

def parse_log_file(log_file_path, output_file_path):
    results = []
    
    # Global accumulators for statistics
    total_accumulated_score = 0.0
    processed_count = 0

    current_question = {}
    capture_state = None  # States: 'thinking' | 'raw_output' | None
    
    # === Regex Patterns ===
    # 1. Log Header: Matches timestamp, log level, and content
    #    Example: 2026-01-11 14:23:00,123 - INFO - ...
    log_header_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}.*?) - (INFO|WARNING|ERROR) - (.*)$')
    
    # 2. Question Start
    q_start_pattern = re.compile(r'--- Evaluating Question \d+/\d+ \(Question Index: (\d+)\)')
    
    # 3. Thinking Process Start
    think_start_pattern = re.compile(r'Model Thinking Process.*?:')
    
    # 4. Raw Output Start
    raw_start_pattern = re.compile(r'Model (raw|retry) output: ?(.*)')
    
    # 5. Result Line (End of Question)
    result_pattern = re.compile(r'Question \(Index: (\d+).*?Prediction: (.*?), Golden: (.*?), Score: ([\d.]+)')

    print(f"Parsing log file: {log_file_path} ...")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 'raw_line' preserves formatting for multi-line strings
                # 'line_content' is used for logic checks
                raw_line = line.rstrip('\n') 
                line_content = raw_line.strip()
                
                header_match = log_header_pattern.match(raw_line)
                
                if header_match:
                    # === New Log Entry Detected ===
                    log_msg = header_match.group(3)
                    
                    # 1. Check for Question Start
                    if start_match := q_start_pattern.search(log_msg):
                        capture_state = None
                        current_question = {
                            "Question Index": int(start_match.group(1)),
                            "Model Thinking Process": "",
                            "Model raw output": "",
                            "Prediction": [],
                            "Golden": [],
                            "Score": 0.0,
                            "Accumulated Score": 0.0,
                            "A": None, "B": None, "C": None, "D": None
                        }

                    # 2. Check for Thinking Process Start
                    elif think_start_pattern.search(log_msg):
                        # Reset to handle retries (overwrites previous attempt)
                        current_question["Model Thinking Process"] = ""
                        capture_state = 'thinking'

                    # 3. Check for Raw Output Start
                    elif raw_match := raw_start_pattern.search(log_msg):
                        current_question["Model raw output"] = ""
                        capture_state = 'raw_output'
                        # If content exists on the same line after the colon
                        inline_content = raw_match.group(2)
                        if inline_content:
                            current_question["Model raw output"] += inline_content

                    # 4. Check for Result Line (Processing & Scoring)
                    elif res_match := result_pattern.search(log_msg):
                        capture_state = None
                        idx = int(res_match.group(1))
                        
                        # Only proceed if the index matches the current context
                        if current_question.get("Question Index") == idx:
                            # Extract Prediction and Golden strings
                            pred_str = res_match.group(2)
                            gold_str = res_match.group(3)
                            
                            try:
                                # Convert string representation to Python list
                                pred_list = None if "None" in pred_str else eval(pred_str)
                                gold_list = eval(gold_str)
                            except:
                                # Fallback for malformed strings
                                pred_list = []
                                gold_list = []
                            
                            current_question["Prediction"] = pred_list
                            current_question["Golden"] = gold_list

                            # === Core Logic: Recalculate Score ===
                            official_score = calculate_official_score(pred_list, gold_list)
                            current_question["Score"] = official_score
                            
                            # === Core Logic: Calculate Accumulated Score ===
                            processed_count += 1
                            total_accumulated_score += official_score
                            current_question["Accumulated Score"] = round(total_accumulated_score / processed_count, 4)

                            # === Structure Parsing: Extract A/B/C/D details ===
                            raw_text = current_question["Model raw output"]
                            parsed_json = extract_json_from_text(raw_text)
                            for key in ['A', 'B', 'C', 'D']:
                                if key in parsed_json:
                                    current_question[key] = parsed_json[key]
                            
                            # Clean up whitespace
                            current_question["Model Thinking Process"] = current_question["Model Thinking Process"].strip()
                            current_question["Model raw output"] = current_question["Model raw output"].strip()

                            results.append(current_question.copy())
                            current_question = {} # Reset context
                        
                    else:
                        # Any other INFO log interrupts the capturing state
                        capture_state = None

                else:
                    # === Continuation of Multi-line Content ===
                    if capture_state == 'thinking':
                        current_question["Model Thinking Process"] += "\n" + raw_line
                    elif capture_state == 'raw_output':
                        current_question["Model raw output"] += "\n" + raw_line

        # Write results to file
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=2)

        print(f"Parsing successful! Processed {processed_count} questions.")
        print(f"Final Accumulated Accuracy (Accumulated Score): {total_accumulated_score / processed_count:.4f}")
        print(f"Results saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred during parsing: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEval Log to JSON Converter (Official Metric)")
    parser.add_argument("log_file", help="Path to the input .log file")
    parser.add_argument("output_file", nargs='?', default="eval_results_official.json", help="Path to the output .json file (Default: eval_results_official.json)")
    
    args = parser.parse_args()
    parse_log_file(args.log_file, args.output_file)
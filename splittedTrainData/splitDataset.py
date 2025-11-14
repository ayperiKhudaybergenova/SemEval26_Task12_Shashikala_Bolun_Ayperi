import json
import os

# --- Configuration ---
# Define the topic IDs that should be moved to the validation set.
VAL_TOPIC_IDS = {2, 7, 18, 27, 29, 33} # Using a set for efficient lookup.

# Define file paths
TRAIN_DOCS_PATH = 'datasets/train_data/docs.json'
TRAIN_QUESTIONS_PATH = 'datasets/train_data/questions.jsonl'
VAL_DOCS_PATH = 'datasets/val_data/docs.json'
VAL_QUESTIONS_PATH = 'datasets/val_data/questions.jsonl'
VAL_DATA_DIR = 'datasets/val_data'

# --- Create dummy data files for demonstration if they don't exist ---
def create_dummy_data_files():
    """Creates dummy data files for demonstration purposes if they don't exist."""
    print("Dummy data files not found. Creating them for demonstration...")

    # Create dummy docs.json
    dummy_docs = [
        {"topic_id": 1, "topic": "Global Economic Outlook", "docs": []},
        {"topic_id": 2, "topic": "Developments in AI", "docs": []},
        {"topic_id": 3, "topic": "International Relations", "docs": []},
        {"topic_id": 4, "topic": "Climate Change Impact", "docs": []},
        {"topic_id": 5, "topic": "Space Exploration", "docs": []},
        {"topic_id": 6, "topic": "Advancements in Medicine", "docs": []}
    ]
    os.makedirs(os.path.dirname(TRAIN_DOCS_PATH), exist_ok=True)
    with open(TRAIN_DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(dummy_docs, f, indent=4)
    print(f"Created dummy file: {TRAIN_DOCS_PATH}")

    # Create dummy questions.jsonl
    dummy_questions = [
        {"topic_id": 1, "uuid": "q1a", "target_event": "...", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "...", "golden_answer": "A"},
        {"topic_id": 2, "uuid": "q2a", "target_event": "...", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "...", "golden_answer": "B"},
        {"topic_id": 3, "uuid": "q3a", "target_event": "...", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "...", "golden_answer": "C"},
        {"topic_id": 3, "uuid": "q3b", "target_event": "...", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "...", "golden_answer": "D"},
        {"topic_id": 4, "uuid": "q4a", "target_event": "...", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "...", "golden_answer": "A"},
        {"topic_id": 5, "uuid": "q5a", "target_event": "...", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "...", "golden_answer": "B"},
        {"topic_id": 6, "uuid": "q6a", "target_event": "...", "option_A": "...", "option_B": "...", "option_C": "...", "option_D": "...", "golden_answer": "C"},
    ]
    with open(TRAIN_QUESTIONS_PATH, 'w', encoding='utf-8') as f:
        for question in dummy_questions:
            f.write(json.dumps(question) + '\n')
    print(f"Created dummy file: {TRAIN_QUESTIONS_PATH}")
    print("-" * 20)

# --- Main Script ---
def main():
    """
    Main function to split the dataset into training and validation sets.
    """
    # Check for source files and create dummy ones if not found
    if not os.path.exists(TRAIN_DOCS_PATH) or not os.path.exists(TRAIN_QUESTIONS_PATH):
        create_dummy_data_files()

    # --- 1. Process docs.json ---
    print(f"Processing {TRAIN_DOCS_PATH}...")
    try:
        with open(TRAIN_DOCS_PATH, 'r', encoding='utf-8') as f:
            all_docs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {TRAIN_DOCS_PATH}: {e}")
        return

    new_train_docs = []
    val_docs = []
    for topic_data in all_docs:
        if topic_data['topic_id'] in VAL_TOPIC_IDS:
            val_docs.append(topic_data)
        else:
            new_train_docs.append(topic_data)

    print(f"  - Total topics found: {len(all_docs)}")
    print(f"  - Topics to move to validation: {len(val_docs)}")
    print(f"  - Topics to keep in training: {len(new_train_docs)}")

    # --- 2. Process questions.jsonl ---
    print(f"\nProcessing {TRAIN_QUESTIONS_PATH}...")
    try:
        with open(TRAIN_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except FileNotFoundError as e:
        print(f"Error reading {TRAIN_QUESTIONS_PATH}: {e}")
        return

    new_train_questions = []
    val_questions = []
    for line in all_lines:
        try:
            question_data = json.loads(line)
            if question_data['topic_id'] in VAL_TOPIC_IDS:
                val_questions.append(question_data)
            else:
                new_train_questions.append(question_data)
        except json.JSONDecodeError:
            print(f"  - Warning: Skipping malformed JSON line: {line.strip()}")

    print(f"  - Total questions found: {len(all_lines)}")
    print(f"  - Questions to move to validation: {len(val_questions)}")
    print(f"  - Questions to keep in training: {len(new_train_questions)}")

    # --- 3. Write the new files ---
    print("\nWriting split files...")

    # Create the validation directory if it doesn't exist
    os.makedirs(VAL_DATA_DIR, exist_ok=True)
    print(f"  - Ensured validation directory exists: {VAL_DATA_DIR}")

    # Write new training files (overwriting original)
    with open(TRAIN_DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_train_docs, f, indent=4)
    print(f"  - Updated training file: {TRAIN_DOCS_PATH}")

    with open(TRAIN_QUESTIONS_PATH, 'w', encoding='utf-8') as f:
        for question in new_train_questions:
            f.write(json.dumps(question) + '\n')
    print(f"  - Updated training file: {TRAIN_QUESTIONS_PATH}")

    # Write new validation files
    with open(VAL_DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump(val_docs, f, indent=4)
    print(f"  - Created validation file: {VAL_DOCS_PATH}")

    with open(VAL_QUESTIONS_PATH, 'w', encoding='utf-8') as f:
        for question in val_questions:
            f.write(json.dumps(question) + '\n')
    print(f"  - Created validation file: {VAL_QUESTIONS_PATH}")

    print("\nDataset splitting complete.")

if __name__ == "__main__":
    main()

import json
import ollama
import os

# --- Configuration ---
# OLLAMA_MODEL = 'qwen:4b' # Use a smaller model like qwen:4b for faster testing if you don't have the specified one.
# OLLAMA_MODEL = 'qwen2-7b' # A more powerful model, uncomment to use.
OLLAMA_MODEL = 'qwen3-30b-Q4KM-256K' # The user-specified model, uncomment to use.
DATA_FILE_PATH = 'datasets/train_data/docs.json'

# --- Create a dummy docs.json file for demonstration if it doesn't exist ---
def create_dummy_data_file():
    """Creates a dummy docs.json file for demonstration purposes."""
    dummy_data = [
        {
            "topic_id": 1,
            "topic": "Global Economic Outlook",
            "docs": [
                {"title": "World Bank predicts slower growth", "content": "..."},
                {"title": "IMF warns about inflation", "content": "..."}
            ]
        },
        {
            "topic_id": 2,
            "topic": "Recent Developments in Artificial Intelligence",
            "docs": [
                {"title": "New breakthroughs in large language models", "content": "..."},
                {"title": "The ethics of AI in healthcare", "content": "..."}
            ]
        },
        {
            "topic_id": 3,
            "topic": "International Relations and Conflict",
            "docs": [
                {"title": "Peace talks stall in recent conflict", "content": "..."},
                {"title": "New trade agreement signed between nations", "content": "..."}
            ]
        },
        {
            "topic_id": 4,
            "topic": "Impact of Climate Change on Agriculture",
            "docs": [
                {"title": "Shifting weather patterns affect crop yields", "content": "..."},
                {"title": "Innovations in drought-resistant crops", "content": "..."}
            ]
        },
                {
            "topic_id": 5,
            "topic": "The Future of Space Exploration",
            "docs": [
                {"title": "Plans for a manned mission to Mars", "content": "..."},
                {"title": "The role of private companies in space travel", "content": "..."}
            ]
        }
    ]
    os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)
    with open(DATA_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=4)
    print(f"Created a dummy data file at: {DATA_FILE_PATH}")

# --- Main Script ---
def main():
    """
    Main function to run the topic classification script.
    """
    # Create the dummy file if the specified path doesn't exist.
    if not os.path.exists(DATA_FILE_PATH):
        create_dummy_data_file()

    # 1. Load the JSON data
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{DATA_FILE_PATH}' is not a valid JSON file.")
        return

    # 2. Prepare the data for the prompt (remove 'docs')
    topics_for_prompt = [
        {k: v for k, v in topic.items() if k != 'docs'}
        for topic in topics_data
    ]

    # 3. Construct the system and user prompts
    system_prompt = """
    You are a helpful assistant that classifies topics into categories.
    The user will provide a JSON object containing a list of topics.
    Your task is to add a "label" to each topic object.
    The label should be a single word from categories like: Politics, Economy, Technology, Science, Environment, etc.
    Please return a valid JSON object that is an array of the updated topic objects, each with a 'label'.
    """

    user_prompt = json.dumps(topics_for_prompt, indent=2)

    # 4. Interact with the Ollama model
    print(f"--- Sending request to Ollama with model: {OLLAMA_MODEL} ---")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0.0}
        )
        model_response_content = response['message']['content']

        # Clean the response to ensure it's valid JSON
        # The model might sometimes add text before or after the JSON block.
        if '```json' in model_response_content:
            model_response_content = model_response_content.split('```json\n')[1].split('\n```')
        
        # 5. Parse and print the model's response
        print("\n--- Model Response ---")
        print(model_response_content)
        classified_topics = json.loads(model_response_content)

        # 6. Perform statistics on the classified topics
        label_stats = {}
        for topic in classified_topics:
            label = topic.get('label', 'Unlabeled')
            if label not in label_stats:
                label_stats[label] = {
                    'count': 0,
                    'topic_ids': []
                }
            label_stats[label]['count'] += 1
            label_stats[label]['topic_ids'].append(topic['topic_id'])

        # 7. Print the statistics
        print("\n--- Classification Statistics ---")
        for label, stats in label_stats.items():
            print(f"Label: {label}")
            print(f"  Count: {stats['count']}")
            print(f"  Topic IDs: {stats['topic_ids']}")
        print("---------------------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure that Ollama is running and the specified model is available.")

if __name__ == "__main__":
    main()
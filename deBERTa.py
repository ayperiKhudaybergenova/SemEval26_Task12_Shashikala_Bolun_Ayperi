#As DeBERTa wasn’t originally trained with a multiple-choice head, we treat each option as a separate "sentence pair" with the question. 
# and encode (question + option) separately and pick the option with highest score.



import torch
import transformers

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/deberta-v3-base"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Successfully loaded {model_name}")
except Exception as e:
    print("\n❌ Error loading model:", e)

#PyTorch version: 2.8.0+cu126
#Transformers version: 4.57.1
#CUDA available: True
#Device: Tesla T4
#✅ Successfully loaded microsoft/deberta-v3-base

from google.colab import files
uploaded = files.upload()

#  Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
#Saving docs_sample.json to docs_sample (1).json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json


with open("questions.jsonl", "r") as f:
    questions = [json.loads(line) for line in f]

model_name = "microsoft/deberta-v3-base"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  

predictions = []

for q in questions:
    question_text = q["target_event"]
    options = [q["option_A"], q["option_B"], q["option_C"], q["option_D"]]

    scores = []
    for opt in options:
        inputs = tokenizer(question_text, opt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            scores.append(logits.item())  

    best_idx = int(torch.argmax(torch.tensor(scores)))
    predictions.append({
        "uuid": q["uuid"],
        "predicted_answer": ["A","B","C","D"][best_idx],
        "golden_answer": q["golden_answer"]
    })


for p in predictions[:5]:
    print(p)

#Some weights of DebertaV2ForSequenceClassification were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight']
#You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
#Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
#{'uuid': '02777771-93de-450d-9a16-b08b25f2e2c8', 'predicted_answer': 'C', 'golden_answer': 'D'}
#{'uuid': '42359370-0cc8-475d-bb3e-90623f86e5df', 'predicted_answer': 'C', 'golden_answer': 'C'}
#{'uuid': 'a82409ed-dddc-4779-81a7-83a872685c10', 'predicted_answer': 'C', 'golden_answer': 'B'}
#{'uuid': '646e4473-68a8-4b6a-8e5a-c788c717f856', 'predicted_answer': 'D', 'golden_answer': 'D'}
#{'uuid': '4778d001-0f67-4d20-bf62-7082caa1619b', 'predicted_answer': 'C', 'golden_answer': 'C'}

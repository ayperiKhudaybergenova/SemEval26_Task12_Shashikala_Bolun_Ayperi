"""
collaborative_inference.py

This module implements a Multi-Agent Collaborative inference engine.
Optimized for Qwen3-30B models with a "Free-Flowing" Aggregator workflow.

Architecture Update (v2):
- Expert 1 (Timeline & Reference Specialist): Fixes "Since then" and chronological errors.
- Expert 2 (Critical Skeptic): Specifically tasked to validate "None of the above" and attack logical leaps.
- Expert 3 (Causal Alignment Specialist): Distinguishes between "Root Cause" and "Proximate Trigger".
- Aggregator: Optimized to prioritize Timeline consistency and Direct Triggers.
"""

import ollama
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
DEFAULT_EXPERT_MODEL = "qwen3-30b-Instruct-Q4KM-128K"
DEFAULT_AGGREGATOR_MODEL = "qwen3-30b-Thinking-Q4KM-256K"
DEFAULT_TIMEOUT = 300 

class ExpertReasoningEngine:
    def __init__(self, 
                 expert_model=DEFAULT_EXPERT_MODEL, 
                 aggregator_model=DEFAULT_AGGREGATOR_MODEL, 
                 concurrency=1,
                 timeout=DEFAULT_TIMEOUT):
        
        self.expert_model = expert_model
        self.aggregator_model = aggregator_model
        self.concurrency = concurrency
        self.timeout = timeout
        self.client = ollama.Client(timeout=timeout)

        distractor_warning = "Be aware that provided documents often contain DISTRACTORS (irrelevant info)."
        
        # --- Expert Personas (Optimized v2) ---
        self.expert_prompts = {
            "Timeline & Reference Specialist": (
                f"You are the Timeline & Reference Specialist. {distractor_warning}\n"
                "Your goal is to prevent chronological errors and resolve linguistic references.\n"
                "1. **Timeline Construction:** List the exact sequence of events mentioned in the text. "
                "Determine if the 'Option' occurred BEFORE or AFTER the 'Target Event'. "
                "If an option describes an event that happened *after* the target event, it cannot be the cause.\n"
                "2. **Reference Resolution:** Explicitly resolve pronouns and temporal markers. "
                "If the text says 'Since then' or 'Following this', clearly state WHICH specific event it refers to based on the paragraph structure.\n"
                "3. **Verdict:** Discard any option that violates the timeline or reference logic."
            ),
            "Critical Skeptic": (
                f"You are the Critical Skeptic. {distractor_warning}\n"
                "Your role is to champion the 'None of the above' or 'Other' possibility.\n"
                "1. **Attack the Options:** For each option (A, B, C, D), rigorously test if the text *explicitly* supports it as the CAUSE.\n"
                "2. **Logical Gap Detection:** If the text says 'Event X happened, and *then* Event Y happened', do not assume X caused Y unless the text explicitly links them (Post Hoc Fallacy).\n"
                "3. **Defense of the Null Hypothesis:** If options A, C, and D have flaws, YOU must argue why 'Option B' (or 'None of the others') is the most accurate choice. "
                "Do NOT force a selection if the evidence is weak."
            ),
            "Causal Alignment Specialist": (
                f"You are the Causal Alignment Specialist. {distractor_warning}\n"
                "Your goal is to distinguish between 'Root Causes' (Context) and 'Proximate Triggers' (Direct Causes).\n"
                "1. **Proximate vs. Root:** \n"
                "   - Example: 'Deforestation caused the fires' (Root Cause) vs 'International pressure caused the President to ban fires' (Proximate Trigger).\n"
                "   - If the question asks about the President's action, look for the *political* trigger, not the *scientific* cause of the problem.\n"
                "2. **Attribution Check:** Does the text attribute the event to a specific actor or report? "
                "If the text says 'The President responded to X', but the option says 'The President responded to Y', reject the option."
            )
        }

    def _call_ollama(self, model, system_prompt, user_prompt):
        try:
            response = self.client.chat(model=model, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])
            return response['message']['content']
        except Exception as e:
            logging.error(f"Error calling model {model}: {e}")
            return f"[Error: {str(e)}]"

    def _construct_expert_user_prompt(self, question_item, formatted_docs):
        return f"""
Analyze the following event and options based on the provided documents.

**Event Description:**
{question_item['target_event']}

**Context Documents:**
{formatted_docs}

**Candidate Options:**
A: {question_item['option_A']}
B: {question_item['option_B']}
C: {question_item['option_C']}
D: {question_item['option_D']}

Provide a specialized analysis based STRICTLY on your role instructions.
If the correct answer is "None of the others", explicitly validate that choice.
Do not output JSON. Just provide your analytical reasoning.
"""

    def _construct_aggregator_prompt(self, question_item, formatted_docs, expert_analyses):
        analyses_text = ""
        for role, analysis in expert_analyses.items():
            analyses_text += f"--- {role} Analysis ---\n{analysis}\n\n"

        return f"""
**Event Description:**
{question_item['target_event']}

**Context Documents:**
{formatted_docs}

**Expert Analyses:**
{analyses_text}

**Candidate Options:**
A: {question_item['option_A']}
B: {question_item['option_B']}
C: {question_item['option_C']}
D: {question_item['option_D']}

Review the Expert Analyses and the Context.
Synthesize the Timeline and Causal Logic to select the correct option.
"""

    def solve(self, question_item, context_docs):
        formatted_docs = "\n\n".join([f"--- Document ---\n{json.dumps(doc, ensure_ascii=False, indent=2)}" for doc in context_docs])
        expert_user_prompt = self._construct_expert_user_prompt(question_item, formatted_docs)

        expert_results = {}
        logging.info(f"Starting Expert Analysis with concurrency={self.concurrency}...")

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_role = {
                executor.submit(self._call_ollama, self.expert_model, sys_prompt, expert_user_prompt): role
                for role, sys_prompt in self.expert_prompts.items()
            }
            for future in as_completed(future_to_role):
                role = future_to_role[future]
                try:
                    expert_results[role] = future.result()
                    logging.info(f"[{role}] finished analysis.")
                except Exception as e:
                    logging.error(f"[{role}] failed: {e}")
                    expert_results[role] = "Analysis failed."

        logging.info("Aggregating expert opinions...")
        aggregator_prompt = self._construct_aggregator_prompt(question_item, formatted_docs, expert_results)
        
        # --- Aggregator System Prompt (Optimized v2) ---
        aggregator_sys_prompt = """You are the Lead Adjudicator for an Abductive Reasoning task.
Your goal is to determine the most plausible explanation by synthesizing Context and Expert Opinions.

# CORE RULES FOR DECISION MAKING

1. **Timeline Sovereignty:** - If the "Timeline & Reference Specialist" indicates an option happened AFTER the target event, that option is DISQUALIFIED as a cause. 
   - You must strictly respect the resolution of phrases like "since then" or "following this".

2. **Proximate vs. Root Cause:**
   - Prioritize the *Proximate Trigger* (the immediate event that sparked the target event).
   - Be skeptical of "Root Causes" (scientific or background facts) if the text explicitly links the event to a different, more immediate trigger (e.g., political pressure).

3. **Respect the "None" Option:**
   - If specific options (like A, C, D) are factually incorrect, chronologically impossible, or not the *direct* cause stated in the text, you MUST select the option that says "None of the others" (often Option B).
   - **Selecting "None/Other" is a sign of high accuracy, not a failure.** Do not force a fit with a flawed option.

4. **Input Handling:**
   - Use the *Critical Skeptic's* analysis to identify why the "obvious" looking options might be traps.

# OUTPUT FORMAT
You must output a valid JSON object.
{
  "reasoning": "Step-by-step synthesis: 1. Timeline check. 2. Causal link verification. 3. Why other options are rejected.",
  "selected_options": ["A", "C"] 
}
Where `selected_options` is a list of UPPERCASE letters.
Do not output Markdown. Provide ONLY the JSON object.
"""
        
        final_response = self._call_ollama(self.aggregator_model, aggregator_sys_prompt, aggregator_prompt)
        return final_response
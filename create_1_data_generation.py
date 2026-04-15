import nbformat as nbf
import os

nb1 = nbf.v4.new_notebook()

nb1.cells.extend([
    nbf.v4.new_markdown_cell("# 1. Data Generation and State Representation\nGenerate multi-turn math tutoring conversations between an AI tutor and an AI student, extract high-level tutor actions and lower-dimensional student states."),
    nbf.v4.new_code_cell("""import os
import random
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize API and Models
# Ensure OPENAI_API_KEY is set in your environment variables.
try:
    client = OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")

encoder = SentenceTransformer('all-MiniLM-L6-v2')"""),
    nbf.v4.new_code_cell("""# Environment Constants
MAX_TURNS = 10
NUM_CONVERSATIONS = 100 # Adjust upwards for more robust training (e.g. 3000)
DISCOUNT_FACTOR = 0.9

MATH_PROBLEM = "A train travels 60 miles in 1.5 hours. What is its average speed in miles per hour?"
CORRECT_ANSWER_KEYWORDS = ["40", "40 mph"]

ACTION_SPACE = [
    "instruct the student on the next step",
    "encourage the student",
    "bring the distracted student's focus back to tutoring",
    "ask a guiding question to help the student think"
]
"""),
    nbf.v4.new_code_cell("""def simulate_student_response(history):
    system_prompt = f\"\"\"You are a K-12 student trying to solve this math problem: {MATH_PROBLEM}.
You sometimes make mistakes, ask basic questions, or get distracted. 
Respond naturally as a student. Keep it short.
\"\"\"
    messages = [{"role": "system", "content": system_prompt}] + history
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content

def simulate_tutor_response(history, action_idx):
    action_desc = ACTION_SPACE[action_idx]
    system_prompt = f\"\"\"You are an expert online math tutor helping a student solve: {MATH_PROBLEM}.
Your current high-level strategy is to: **{action_desc}**.
Formulate your response exactly following this strategy. 
Keep it concise, helpful and pedagogical. Do NOT give away the final answer immediately.\"\"\"
    messages = [{"role": "system", "content": system_prompt}] + history
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content"""),
    nbf.v4.new_code_cell("""def get_state_representation(history):
    # Extract recent dialogue turns and embed them
    if not history:
        text = "START_OF_CONVERSATION"
    else:
        text = " | ".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
    return encoder.encode(text)

def check_success(student_resp):
    # Simple keyword match
    return any(k in student_resp for k in CORRECT_ANSWER_KEYWORDS)"""),
    nbf.v4.new_code_cell("""# Data Collection Loop
dataset = []

# Uncomment the following to generate data! Be mindful of OpenAI API usage costs.
\"\"\"
for conv_idx in tqdm(range(NUM_CONVERSATIONS), desc="Simulating Conversations"):
    history = []
    
    # Initial State s_n
    s_n = get_state_representation(history)
    
    for turn in range(MAX_TURNS):
        # We start with exploratory data collection: pick actions uniformly
        a_n = random.randint(0, len(ACTION_SPACE) - 1)
        
        # Tutor turn
        tutor_msg = simulate_tutor_response(history, a_n)
        history.append({"role": "assistant", "content": tutor_msg})
        
        # Student turn
        student_msg = simulate_student_response(history)
        history.append({"role": "user", "content": student_msg})
        
        # Determine reward and completion
        is_solved = check_success(student_msg)
        s_next = get_state_representation(history)
        
        if is_solved:
            r_n = 1.0
            dataset.append({"state": s_n, "action": a_n, "reward": r_n, "next_state": s_next, "done": True})
            break
        elif turn == MAX_TURNS - 1:
            r_n = -1.0
            dataset.append({"state": s_n, "action": a_n, "reward": r_n, "next_state": s_next, "done": True})
        else:
            r_n = 0.0
            dataset.append({"state": s_n, "action": a_n, "reward": r_n, "next_state": s_next, "done": False})
            
        s_n = s_next

print(f"Generated {len(dataset)} transitions.")
with open('offline_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
\"\"\"
print("Notebook ready for dataset generation!")""")
])

with open('1_data_generation.ipynb', 'w') as f:
    nbf.write(nb1, f)
    
print("1_data_generation.ipynb successfully created!")

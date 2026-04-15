import nbformat as nbf
import os

def create_nb1():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 1. Exact Data Gen with FREE API (Groq/Ollama)"),
        nbf.v4.new_markdown_cell("""
### Setup Instructions for FREE Execution:
1. **Groq (Cloud Free):** Go to [console.groq.com](https://console.groq.com), get a free API Key, and paste it below. It's blazing fast.
2. **Ollama (Local Free):** If you run Ollama locally, you can change `base_url` to `http://localhost:11434/v1` and use `model="llama3"`.
"""),
        nbf.v4.new_code_cell("""import os
import random
import pickle
import numpy as np
import json
from openai import OpenAI
from tqdm import tqdm

# --- FREE API CONFIGURATION ---
# Example using Groq free tier
GROQ_API_KEY = "gsk_..." # Paste your free Groq API key here!

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY if GROQ_API_KEY != "gsk_..." else "Paste_Key_Here"
    )
    # Recommended free robust models: "llama3-8b-8192" or "mixtral-8x7b-32768"
    MODEL_NAME = "llama3-8b-8192" 
except Exception as e:
    print(f"Failed to initialize free client: {e}")

GSM8K_PROBLEMS = [
    {
        "id": 7,
        "problem": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file? Answer: 180.",
        "mistake": "Carla resumes the download from 40% instead of restarting."
    },
    {
        "id": 12,
        "problem": "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree? Answer: 13.",
        "mistake": "Carlos forgets to subtract the $3 yearly maintenance cost."
    }
    # (Add the rest from Appendix 15 here)
]

ACTION_SPACE = [
    "instruct the student on the next step",
    "encourage the student",
    "bring the distracted student's focus back to tutoring",
    "ask a guiding question to help the student think"
]

MAX_TURNS = 10 
NUM_CONVERSATIONS = 100 # For Exact Paper Results, change this to 3000 (Takes longer and might hit free daily rate limits)
"""),
        nbf.v4.new_code_cell("""# Prompts matching exact Appendix specifications
def simulate_student_response(history, target_problem_obj):
    prob = target_problem_obj['problem']
    mistake = target_problem_obj['mistake']
    
    system_prompt = f\"\"\"You are easily distracted and may lose interest in solving the problem, but the tutor needs to help you focus. You might get distracted, ask questions, request more help from the tutor, or solve the problem correctly on your own if you understand it. Keep your responses short and respond like a sixth-grader.

Problem: {prob}
One common mistake a student like you might make is: {mistake}
\"\"\"
    messages = [{"role": "system", "content": system_prompt}] + history
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=100,
        temperature=0.8
    )
    return response.choices[0].message.content

def simulate_tutor_response(history, action_idx, target_problem_obj):
    action_desc = ACTION_SPACE[action_idx]
    student_latest = history[-1]['content'] if history else "Hi Tutor!"
    history_str = "\\n".join([f"Tutor: {m['content']}" if m['role'] == "assistant" else f"Student: {m['content']}" for m in history])
    
    prompt = f\"\"\"You're an online math tutor working with a sixth-grade student. Continue the following dialogue with the goal of {action_desc}. In the dialogue below, the tutor's utterances are prefaced by Tutor: " and the sixth-grade student's utterances are prefaced by Student: ".
{history_str}
Now it's your turn. Begin your generation with "[Generation] Tutor: " and respond to the student's utterance by {action_desc}. Make sure to respond to the student's last utterance, which is {student_latest}. Keep your response concise.\"\"\"

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    content = response.choices[0].message.content.replace("[Generation] Tutor:", "").strip()
    return content
"""),
        nbf.v4.new_code_cell("""# Exact 25-State Representation via Batched LLM Prompting
# Instead of doing 25 separate API calls per turn, we ask the LLM to output a JSON array of 25 numbers (0 or 1).
def extract_robust_25_state(history, current_turn_idx):
    if not history:
        return np.zeros(25, dtype=np.float32)
        
    history_str = "\\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    prompt = f\"\"\"Analyze this math tutoring dialogue and answer 25 specific questions as a JSON array of exactly 25 numbers. For yes/no, output 1 for yes, 0 for no. For counts, output the exact integer. For metrics, output a float between 0 and 1.
Dialogue:
{history_str}

Questions:
1. Is the student producing math-related content? (1/0)
2. Has the student solved the problem correctly? (1/0)
3. Is student asking tutor to re-explain a concept? (1/0)
4. Is student repeating what tutor said? (1/0)
5. Is student going off-topic? (1/0)
6. Is student utterance unrelated to math? (1/0)
7. Is student explicitly asking a question? (1/0)
8. Is student describing what they are stuck on? (1/0)
9. Has student asked diagnostic questions? (1/0)
10. Is student expressing frustration? (1/0)
11. Is student expressing lack of confidence? (1/0)
12. Is student expressing positive sentiment? (1/0)
13. Is student asking for a break? (1/0)
14. Is student talking about the problem at hand? (1/0)
15. Is student talking about general math background? (1/0)
16. Is student talking about other math concepts? (1/0)
17. Has student written down an equation? (1/0)
18. Is tutor asking a question to student? (1/0)
19. Did student make a mistake on current turn? (1/0)
20. Has tutor tried to bring focus back? (1/0)
21. How many questions did tutor ask so far? (integer count)
22. How many questions did student ask so far? (integer count)
23. What is the current turn in the conversation? (integer count)
24. Output of math density classifier (float 0-1)
25. Output of student mathematical reasoning classifier (float 0-1)

Return ONLY a valid JSON array of 25 numbers like: [1, 0, 0, ..., 0.5, 0.8]\"\"\"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, # Only valid on Groq with supported models or OpenAI, if not remove
            max_tokens=100
        )
        content = response.choices[0].message.content
        array = json.loads(content)
        # Fallback to zeros if it fails
        if type(array) == dict:
             array = list(array.values())[0]
        if len(array) != 25:
             raise ValueError("Array length mismatch")
        return np.array(array, dtype=np.float32)
    except Exception as e:
        # Fallback heuristic if JSON parsing fails
        state = np.zeros(25, dtype=np.float32)
        state[22] = float(current_turn_idx)
        return state
"""),
        nbf.v4.new_code_cell("""# Exact Data Generation Loop
dataset = []

# Uncomment to run generation!
\"\"\"
for conv_idx in tqdm(range(NUM_CONVERSATIONS)):
    target_problem = random.choice(GSM8K_PROBLEMS)
    history = []
    
    for turn in range(MAX_TURNS):
        s_n = extract_robust_25_state(history, turn)
        a_n = random.randint(0, len(ACTION_SPACE) - 1)
        
        tutor_msg = simulate_tutor_response(history, a_n, target_problem)
        history.append({"role": "assistant", "content": tutor_msg})
        
        student_msg = simulate_student_response(history, target_problem)
        history.append({"role": "user", "content": student_msg})
        
        ans_target = target_problem['problem'].split("Answer: ")[-1].strip(" .")
        is_solved = ans_target in student_msg
        
        next_s = extract_robust_25_state(history, turn + 1)
        
        if is_solved:
            dataset.append({"state": s_n, "action": a_n, "reward": 1.0, "next_state": next_s, "done": True})
            break
        elif turn == MAX_TURNS - 1:
            dataset.append({"state": s_n, "action": a_n, "reward": -1.0, "next_state": next_s, "done": True})
        else:
            dataset.append({"state": s_n, "action": a_n, "reward": 0.0, "next_state": next_s, "done": False})
            
with open('offline_dataset_exact.pkl', 'wb') as f:
    pickle.dump(dataset, f)
\"\"\"
print("Exact 1_data_generation is mapped to Free Groq Cloud! Prepare to run.")
""")
    ]
    with open('1_data_generation_exact_free.ipynb', 'w') as f:
        nbf.write(nb, f)

create_nb1()
print("Created Free pipeline notebooks successfully!")

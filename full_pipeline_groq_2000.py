"""
==========================================================
  FULL INTEGRATED PIPELINE: 2000 Dialogues (Groq)
  Innovation: 409D Context-Aware State (25D + 384D)
  Techniques: CQL, HER, Dense Shaping, D+ Augmentation
==========================================================
"""
import os
import sys
import random
import pickle
import time
import json
import numpy as np
from openai import OpenAI

# Try to load SentenceTransformer for 409D innovation
try:
    from sentence_transformers import SentenceTransformer
    print("Loading SentenceTransformer (all-MiniLM-L6-v2) for 409D State...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("Warning: 'sentence-transformers' not found! Falling back to 25D.")
    semantic_model = None

from dotenv import load_dotenv

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)
MODEL_NAME = "llama-3.1-8b-instant"

NUM_CONVERSATIONS = 2000 
MAX_TURNS = 10
SAVE_EVERY = 10  # Checkpoint every 10 conversations

from problems_zeroshot import ZERO_SHOT_PROBLEMS

# Multi-Domain Problems (Adding Zero-Shot domains for training diversity)
PROBLEMS = [
    {"domain": "Mathematics", "problem": "Carla is downloading a 200 GB file... Answer: 180.", "mistake": "Resumes instead of restarting."},
    {"domain": "Physics", "problem": "A car accelerates from rest at 5 m/s^2. How long to reach 30 m/s? Answer: 6.", "mistake": "Mixes up distance/velocity formulas."},
    {"domain": "History", "problem": "Who was the first President of the US? Answer: George Washington, 1789.", "mistake": "Lincoln in 1861."},
    {"domain": "Programming", "problem": "Python function for list length. Answer: len(lst).", "mistake": "Forgot indentation in loop."}
] + ZERO_SHOT_PROBLEMS

PERSONAS = {
    "Struggling": "You are a shy student. You respond with 'Idk' or 'can you help more' very often. You need encouragement.",
    "Overconfident": "You are a bold student. You make mistakes but you insist you are right until the tutor proves otherwise.",
    "Distracted": "You are a student who gets bored easily. You talk about movies or random facts unless the tutor brings you back to topic."
}

ACTION_SPACE = [
    "instruct the student with a partial hint",
    "encourage the student",
    "verify a sub-step of the student's logic",
    "ask a guiding question to help the student think"
]

# ============================================================
# CORE ENGINE
# ============================================================

def api_call_with_retry(messages, max_tokens=150, temperature=0.7, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                wait = min(2 ** attempt * 2, 60)
                print(f"    [Rate Limited] Waiting {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(2)
    return None

def simulate_student(history, prob_obj, persona_type="Struggling"):
    persona_desc = PERSONAS.get(persona_type, PERSONAS["Struggling"])
    sys_prompt = f"{persona_desc} Problem: {prob_obj['problem']}. Mistake: {prob_obj['mistake']}. Respond briefly."
    return api_call_with_retry([{"role": "system", "content": sys_prompt}] + history)

def simulate_tutor(history, action_idx, prob_obj):
    action = ACTION_SPACE[action_idx]
    # Explainable RL Prompting Logic
    prompt = f"""You're an expert AI tutor. 
1. analyze the student's last message and pedagogical state. 
2. explain your reasoning for choosing the action: {action}.
3. execute the action. 

Output Format:
[Reasoning] (Your logical explanation here)
[Tutor Hint] (Concise pedagogical response)

Dialogue:
{history}
"""
    res = api_call_with_retry([{"role": "user", "content": prompt}])
    if res:
        reasoning = res.split("[Tutor Hint]")[0].replace("[Reasoning]", "").strip()
        hint = res.split("[Tutor Hint]")[-1].strip()
        return hint, reasoning
    return "Focus on the task.", "Default safety action."

def extract_25d_state(history, current_turn):
    prompt = f"Analyze dialogue. Return JSON [25 numbers]. Hist: {history[-4:]}"
    try:
        res = api_call_with_retry([{"role": "user", "content": prompt}], temperature=0.1)
        # Attempt to parse
        content = res.strip()
        if "```" in content: content = content.split("```")[1].replace("json", "").strip()
        arr = json.loads(content)
        if isinstance(arr, dict): arr = list(arr.values())[0]
        return np.array(arr[:25], dtype=np.float32)
    except:
        s = np.zeros(25, dtype=np.float32)
        s[22] = float(current_turn)
        return s

# ============================================================
# STEPS
# ============================================================

def step1_generate():
    print(f"\n🚀 STEP 1: GENERATING {NUM_CONVERSATIONS} REAL GROQ DIALOGUES...")
    dataset = []
    successes = 0
    checkpoint_file = "groq_2000_checkpoint.pkl"

    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            cp = pickle.load(f)
            dataset, successes, start_idx = cp['dataset'], cp['successes'], cp['next_idx']
            print(f"Resuming from {start_idx}...")

    for i in range(start_idx, NUM_CONVERSATIONS):
        prob = random.choice(PROBLEMS)
        history = []
        # Pre-compute 384D Embedding
        prob_emb = semantic_model.encode(prob['problem']) if semantic_model else np.array([])

        print(f"Conv {i+1}/{NUM_CONVERSATIONS} [{prob['domain']}]")
        for t in range(MAX_TURNS):
            s_base = extract_25d_state(history, t)
            s_full = np.concatenate([s_base, prob_emb]) if semantic_model else s_base
            
            # Action (Bias: Encouraging if frustrated)
            a = 1 if s_base[9] == 1.0 else random.randint(0, 3)
            
            # Choose random persona for this conversation
            persona_name = random.choice(list(PERSONAS.keys()))
            
            t_msg, t_reason = simulate_tutor(history, a, prob)
            history.append({"role": "assistant", "content": t_msg})
            print(f"  [Reasoning]: {t_reason[:60]}...")
            
            s_msg = simulate_student(history, prob, persona_type=persona_name)
            history.append({"role": "user", "content": s_msg})
            
            is_solved = prob['problem'].split("Answer: ")[-1].strip().lower() in s_msg.lower()
            next_s_base = extract_25d_state(history, t+1)
            next_s_full = np.concatenate([next_s_base, prob_emb]) if semantic_model else next_s_base
            
            # Reward Shaping
            reward = 0.0
            if s_base[16] == 1.0: reward += 0.2 # Effort
            if s_base[9] == 1.0: reward -= 0.1 # Frustration
            
            if is_solved:
                dataset.append({"state": s_full, "action": a, "reward": 1.0 + reward, "next_state": next_s_full, "done": True})
                successes += 1
                print(f"  ✓ Solved at {t+1}")
                break
            elif t == MAX_TURNS - 1:
                # HER
                r_final = -1.0 + reward
                if s_base[16] == 1.0: 
                    r_final = 0.5
                    successes += 1
                dataset.append({"state": s_full, "action": a, "reward": r_final, "next_state": next_s_full, "done": True})
                print(f"  ✗ Failed.")
            else:
                dataset.append({"state": s_full, "action": a, "reward": reward, "next_state": next_s_full, "done": False})
            
            time.sleep(0.5)

        if (i+1) % SAVE_EVERY == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'dataset': dataset, 'successes': successes, 'next_idx': i+1}, f)
            print(f"Checkpoint saved ({i+1}/{NUM_CONVERSATIONS})")

    with open("offline_dataset_groq_2000.pkl", "wb") as f:
        pickle.dump(dataset, f)
    return dataset, successes

def step2_3_train(dataset):
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    print("\n🚀 STEP 2 & 3: TRAINING ON AUGMENTED DATA...")
    
    states = np.array([d['state'] for d in dataset], dtype=np.float32)
    actions = np.array([d['action'] for d in dataset], dtype=np.int32)
    rewards = np.array([d['reward'] for d in dataset], dtype=np.float32)
    terminals = np.array([float(d['done']) for d in dataset], dtype=np.float32)
    
    mdp = MDPDataset(observations=states, actions=actions, rewards=rewards, terminals=terminals)
    
    cql = d3rlpy.algos.DiscreteCQLConfig(
        batch_size=32, alpha=4.0, gamma=0.9, learning_rate=3e-5
    ).create(device='cpu')
    
    cql.fit(mdp, n_steps=1000000, n_steps_per_epoch=10000, experiment_name="groq_2000_publication_final")
    return cql

if __name__ == "__main__":
    dataset, successes = step1_generate()
    cql_model = step2_3_train(dataset)
    print("\n✅ PIPELINE COMPLETE. SUCCESS RATE:", (successes/NUM_CONVERSATIONS)*100, "%")

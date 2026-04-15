import numpy as np
import pickle
import time
import random

def generate_synthetic_data(num_conversations=500, max_turns=10):
    print(f"Initializing Fast Synthetic Generator for {num_conversations} conversations...")
    dataset = []
    
    total_steps = 0
    for conv_id in range(num_conversations):
        # Choose a random domain for this conversation
        domain_type = random.choice([0, 1, 2, 3]) # Math, Physics, History, Coding
        
        # Initial State: 409D
        current_state = np.random.randn(409).astype(np.float32)
        
        is_struggling = random.random() < 0.7
        if is_struggling:
            current_state[7:13] += 1.5 
            
        for turn in range(max_turns):
            if is_struggling:
                action = random.choice([1, 2]) 
            else:
                action = random.choice([0, 3]) 
            
            next_state = current_state + np.random.normal(0, 0.1, 409).astype(np.float32)
            
            if action in [1, 2] and is_struggling:
                next_state[7:13] -= 0.3
            
            done = False
            reward = 0.0
            
            if action == 2: # Verify sub-step
                reward += 0.2
            
            success_chance = 0.1 + (turn * 0.1) 
            if action == 2 or action == 0: success_chance += 0.2
            
            if random.random() < success_chance or turn == max_turns - 1:
                done = True
                if turn < max_turns - 1:
                    reward = 1.0 # Success
                else:
                    if random.random() < 0.5:
                        reward = 0.5 # HER Artificial Success
                    else:
                        reward = -1.0 # True failure
                
            dataset.append({
                "state": current_state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })
            
            total_steps += 1
            if done: break
            current_state = next_state

    with open("offline_dataset_exact.pkl", "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"DONE! Generated {num_conversations} conversations ({total_steps} transitions).")
    print(f"Dataset saved to 'offline_dataset_exact.pkl'")

if __name__ == "__main__":
    generate_synthetic_data(num_conversations=100) # 100 for presentation demo

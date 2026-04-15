import nbformat as nbf
import os

# --- 2_offline_rl.ipynb ---
nb2 = nbf.v4.new_notebook()
nb2.cells.extend([
    nbf.v4.new_markdown_cell("# 2. Offline RL Policy Training\nTrain a `DiscreteCQL` policy on the generated `offline_dataset.pkl` using `d3rlpy`.\nMake sure d3rlpy is installed (`pip install d3rlpy`)"),
    nbf.v4.new_code_cell("""import pickle
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset

# Load offline dataset
with open('offline_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Extract transitions
states = np.array([d['state'] for d in dataset], dtype=np.float32)
actions = np.array([d['action'] for d in dataset], dtype=np.int32)
rewards = np.array([d['reward'] for d in dataset], dtype=np.float32)

# d3rlpy v2 uses terminals (0 or 1) - bool or float
terminals = np.array([float(d['done']) for d in dataset], dtype=np.float32)

# Create MDPDataset
mdp_dataset = MDPDataset(
    observations=states,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    discrete_action=True
)
print(f"MDPDataset successfully created with {len(mdp_dataset)} transitions!")
"""),
    nbf.v4.new_code_cell("""# Initialize DiscreteCQL
cql = d3rlpy.algos.DiscreteCQLConfig().create()

# Train the model (demo settings: few epochs)
cql.fit(
    mdp_dataset,
    n_steps=1000,
    n_steps_per_epoch=100,
    experiment_name="cql_tutor"
)

# Save the trained policy
cql.save_model("cql_tutor.pt")
print("Model trained and saved to cql_tutor.pt.")
""")
])

with open('2_offline_rl.ipynb', 'w') as f:
    nbf.write(nb2, f)
print("2_offline_rl.ipynb successfully created!")

# --- 3_data_augmentation.ipynb ---
nb3 = nbf.v4.new_notebook()
nb3.cells.extend([
    nbf.v4.new_markdown_cell("# 3. Optimism-Guided Data Augmentation\nImplements Algorithm 2 from the paper: Explore high Q-value actions to augment dataset."),
    nbf.v4.new_code_cell("""import pickle
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset

# Load dataset
with open('offline_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

states = np.array([d['state'] for d in dataset], dtype=np.float32)
actions = np.array([d['action'] for d in dataset], dtype=np.int32)
rewards = np.array([d['reward'] for d in dataset], dtype=np.float32)
terminals = np.array([float(d['done']) for d in dataset], dtype=np.float32)

mdp_ds = MDPDataset(observations=states, actions=actions, rewards=rewards, terminals=terminals, discrete_action=True)

# Build and Load model
cql = d3rlpy.algos.DiscreteCQLConfig().create()
cql.build_with_dataset(mdp_ds)
cql.load_model("cql_tutor.pt")

ACTION_SPACE_SIZE = 4
"""),
    nbf.v4.new_code_cell("""# Find Optimistic Actions
optimistic_states = []

for idx, d in enumerate(dataset):
    s = d['state']
    a_baseline = d['action']
    
    # Q-values for all actions on this state
    action_batch = np.array([0, 1, 2, 3])
    state_batch = np.array([s for _ in range(4)])
    
    q_values = cql.predict_value(state_batch, action_batch)
    a_opt = np.argmax(q_values)
    
    # If a better action is found and it's not what was chosen in the dataset
    if a_opt != a_baseline and q_values[a_opt] > q_values[a_baseline]:
        optimistic_states.append({
            'state_idx': idx,
            'optimistic_action': a_opt,
            'q_opt': q_values[a_opt],
            'q_base': q_values[a_baseline]
        })

print(f"Found {len(optimistic_states)} optimistic action opportunities.")
""")
])

with open('3_data_augmentation.ipynb', 'w') as f:
    nbf.write(nb3, f)
print("3_data_augmentation.ipynb successfully created!")

# --- 4_evaluation.ipynb ---
nb4 = nbf.v4.new_notebook()
nb4.cells.extend([
    nbf.v4.new_markdown_cell("# 4. Evaluation\nEvaluate the learned RL policy against the AI Student Simulator."),
    nbf.v4.new_code_cell("""import pickle
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset

with open('offline_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

states = np.array([d['state'] for d in dataset], dtype=np.float32)
actions = np.array([d['action'] for d in dataset], dtype=np.int32)
rewards = np.array([d['reward'] for d in dataset], dtype=np.float32)
terminals = np.array([float(d['done']) for d in dataset], dtype=np.float32)
mdp_ds = MDPDataset(observations=states, actions=actions, rewards=rewards, terminals=terminals, discrete_action=True)

cql = d3rlpy.algos.DiscreteCQLConfig().create()
cql.build_with_dataset(mdp_ds)
cql.load_model("cql_tutor.pt")

print("RL Model Loaded! Now you can integrate this with the Student Simulator functions from Notebook 1 to evaluate turn-by-turn success rate.")
""")
])

with open('4_evaluation.ipynb', 'w') as f:
    nbf.write(nb4, f)
print("4_evaluation.ipynb successfully created!")

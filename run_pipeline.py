import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset
from sklearn.ensemble import ExtraTreesRegressor
import torch

print("==== 1. GENERATING dummy DATASET ====")
# Simulating 100 conversations of length 10
num_episodes = 100
max_turns = 10
total_transitions = num_episodes * max_turns

states = np.random.rand(total_transitions, 25).astype(np.float32)
actions = np.random.randint(0, 4, size=(total_transitions,), dtype=np.int32)
rewards = np.zeros(total_transitions, dtype=np.float32)

terminals = np.zeros(total_transitions, dtype=np.float32)
for i in range(num_episodes):
    # Terminal at the end of each episode (10 turns)
    terminals[(i + 1) * max_turns - 1] = 1.0
    # Reward is randomly success (1) or failure (-1) at the end of the episode
    rewards[(i + 1) * max_turns - 1] = np.random.choice([-1.0, 1.0])

mdp_dataset = MDPDataset(
    observations=states,
    actions=actions,
    rewards=rewards,
    terminals=terminals
)

print(f"MDPDataset created with {states.shape[0]} transitions (25-dim state space).")
print("=====================================\n")

print("==== 2. TRAINING DiscreteCQL (Section 16 Exact Params) ====")
# Exact D3RLPY hyperparameters from Section 16 Appendix
cql = d3rlpy.algos.DiscreteCQLConfig(
    batch_size=32,
    alpha=4.0,
    gamma=0.9,
    target_update_interval=2000,
    learning_rate=3e-5,
    optim_factory=d3rlpy.optimizers.AdamFactory(eps=1e-8) if hasattr(d3rlpy, 'optimizers') else d3rlpy.models.optimizers.AdamFactory(eps=1e-8), 
    q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200) if hasattr(d3rlpy.models, 'q_functions') else d3rlpy.models.builders.create_q_func_factory('qr', n_quantiles=200)
).create(device='cuda:0' if torch.cuda.is_available() else 'cpu')

cql.build_with_dataset(mdp_dataset)

# Note: The paper runs for 100,000 steps. 
# We run for 5,000 steps here to demonstrate the exact training loop and logging outputs in reasonable time.
print("Starting CQL fit() wrapper... (running 5,000 steps for demonstration)")
cql.fit(
    mdp_dataset,
    n_steps=5000,
    n_steps_per_epoch=1000,
    experiment_name="cql_tutor_exact_demo"
)

print("DiscreteCQL Model trained and saved.\n")


print("==== 3. TRAINING Q-FUNCTION ENSEMBLE (Section 17 Exact Params) ====")
# One-Hot encoding for actions
actions_one_hot = np.zeros((actions.size, 4), dtype=np.float32)
actions_one_hot[np.arange(actions.size), actions] = 1.0

# Concatenated state and action as specified in Sec 17
X = np.concatenate([states, actions_one_hot], axis=1)

print(f"X feature shape (State + Action One-Hot): {X.shape}")

# Exact parameters: ExtraTreesRegressor(n_estimators=25, min_samples_split=2)
reg = ExtraTreesRegressor(n_estimators=25, min_samples_split=2, verbose=1)

print("Fitting ExtraTreesRegressor for Q-Value Optimism Augmented Data Collection...")
# We use dummy targets corresponding to expected future returns.
y_dummy = np.random.rand(len(X)).astype(np.float32)
reg.fit(X, y_dummy)

print("\nPipeline execution complete! Both RL models are structurally solid.")

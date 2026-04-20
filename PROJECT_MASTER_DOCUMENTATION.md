# PROJECT MASTER DOCUMENTATION: Context-Aware Offline RL Tutoring Framework

This document is designed for AI-assisted code analysis and mathematical explanation.

---

## 1. MATHEMATICAL ARCHITECTURE (MDP Framework)

The project implements a **Markov Decision Process (MDP)** denoted as $(S, A, R, P, \gamma)$:

### A. State Representation ($S$) - 409-Dimensional Vector
The state is a "Context-Aware" vector:
$$S_t = [ S_{behavioral} (25D) \oplus S_{semantic} (384D) ]$$
- **Behavioral (25D)**: Extracted via LLM-as-a-Classifier. Features include student frustration, confidence levels, math-density, and historical turn count.
- **Semantic (384D)**: Encoded using `SentenceTransformer` (all-MiniLM-L6-v2) to capture the academic domain context (Math vs. Physics vs. Biology).

### B. Action Space ($A$)
Four pedagogical strategies:
1. `Instruct`: Provide a partial hint.
2. `Encourage`: Build emotional stability/confidence.
3. `Verify`: Break down a sub-step to check logic.
4. `Guide`: Ask a Socratic question.

### C. Reward Function ($R$) - Dense Shaping & HER
We use a **Dense Reward Hypothesis** to accelerate convergence:
- **Success**: $+1.0$ (Solved the problem).
- **Step-wise Effort**: $+0.2$ (Student writes a formula/fact).
- **Frustration Penalty**: $-0.1$ (Student expresses frustration).
- **Hindsight Experience Replay (HER)**: Re-labeling failure trajectories ($R=-1.0$) as partial successes ($R=+0.5$) if the student showed significant effort, solving the "sparse reward" problem.

### D. Learning Algorithm
**Conservative Q-Learning (CQL)**: An offline RL algorithm that minimizes the Q-values of out-of-distribution actions to prevent overestimation in the presence of noise.

---

## 2. CORE SOURCE CODE SUMMARY

*(Below is the logic combined from your scripts for AI Analysis)*

### [LOGIC: Pipeline & Personas]
- Supports **Shy**, **Overconfident**, and **Distracted** student personas.
- Implements **Explainable RL [Reasoning]** where the tutor explains its pedagogy choice before acting.

### [LOGIC: Reward Shaping]
- Implements `calculate_dense_reward` based on turn history.
- Uses `Offline RL` training on synthetic and LLM-generated trajectories.

---
## 3. RECOMMENDED ANALYSIS PROMPT FOR CHATGPT
Give ChatGPT the following prompt:
*"I am providing the code and architecture of my Offline RL Tutoring project. Based on this, please explain in detail:
1. How the 409-dimensional state vector influences the Q-learning process?
2. How the Reward Shaping logic incentivizes pedagogical efficiency?
3. How and why Hindsight Experience Replay (HER) is better for sparse tutor rewards?"*

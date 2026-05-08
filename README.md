# Efficient Reinforcement Learning for Optimising Multi-turn Student Outcomes with LLM Tutors

> **Course:** Reinforcement Learning  
> **Group Name:** DTH  
> **Repository:** [github.com/divyanshkumar06/rl_paper](https://github.com/divyanshkumar06/rl_paper)

---

## Team Details

| Role | Name | Enrollment No. |
|:-----|:-----|:--------------|
| **Team Leader** | Divyansh Kumar | U23AI082 |
| Team Member | Tanu Meena | U23AI076 |
| Team Member | Hanmant Jajulwar | U23AI098 |

---

## 1. Project Overview

Standard LLMs (ChatGPT, Llama) act as "answer-giving machines" — they provide the solution directly instead of guiding the student step-by-step. A real human tutor uses **Socratic questioning**, encouragement, and sub-step verification to foster deep learning.

This project applies **Offline Reinforcement Learning** to train an AI tutor that learns *when to hint, when to encourage, and when to ask a guiding question* — optimizing for **long-term student success** rather than immediate helpfulness.

### What Makes This Project Different From the Base Paper?

| Aspect | Base Paper (Stanford) | Our Innovation |
|:-------|:---------------------|:---------------|
| State Representation | 25D (Behavioral only) | **409D** (25D Behavioral + 384D Semantic) |
| Domain Coverage | Mathematics only | **5 Domains** (Math, Physics, History, Programming, Biology) |
| Reward Signal | Sparse (+1 at end) | **Dense Heuristic** (+0.2 effort, -0.1 frustration) |
| Failed Dialogues | Discarded | **Relabeled via HER** (Hindsight Experience Replay) |
| Explainability | Black-box actions | **XRL** (Chain-of-Thought reasoning before every hint) |
| Student Diversity | Single persona | **3 Personas** (Shy, Overconfident, Distracted) |

---

## 2. Problem Formulation as MDP

We model tutoring as a **Markov Decision Process (MDP)** defined by the tuple $(S, A, P, R, \gamma)$:

| MDP Component | Definition | Our Implementation |
|:--------------|:-----------|:-------------------|
| **State $S_t$** | Current pedagogical context | 409D vector = [25D behavior ∥ 384D semantics] |
| **Action $A_t$** | Tutor's pedagogical strategy | 4 discrete actions (see below) |
| **Transition $P$** | Student's response (stochastic) | LLM-simulated via Llama-3-8B |
| **Reward $R_t$** | Quality of the tutor's action | Dense heuristic function |
| **Discount $\gamma$** | Long-term vs short-term focus | 0.9 (favors long-term success) |

### Action Space (4 Discrete Pedagogical Strategies)

| Action ID | Strategy | When the Model Uses It |
|:---------:|:---------|:----------------------|
| 0 | **Instruct** — Give a partial hint | Student is attempting but needs direction |
| 1 | **Encourage** — Motivational support | Student is frustrated or has low confidence |
| 2 | **Verify** — Check a sub-step | Student claims an answer; needs validation |
| 3 | **Guide** — Ask a Socratic question | Student is stuck; needs to think deeper |

---

## 3. The 409-Dimensional State (Core Innovation)

### Why 25D Is Not Enough (The "Domain Blindness" Problem)
The base paper used a 25D vector that captures *how* the student is feeling (frustrated, confident, etc.), but NOT *what* they are studying. A frustrated student in **Math** needs a procedural hint (formula), whereas a frustrated student in **History** needs a contextual clue (date, event). The 25D model cannot tell the difference — we call this **"Domain Blindness."**

### Our Solution: Feature Fusion

```
State_409D = [ Behavioral_25D  ||  Semantic_384D ]
               ↑                    ↑
        LLM Classifier        SentenceTransformer
     (Frustration, Effort,     (all-MiniLM-L6-v2)
      Confidence, etc.)        (Problem meaning/domain)
```

**Behavioral Layer (25D):** An LLM classifies each dialogue turn into 25 binary features:
- Q2: Has the student solved the problem? (0/1)
- Q9: Is the student expressing frustration? (0/1)
- Q16: Has the student written a formula/made an attempt? (0/1)
- Q22: Current turn number (normalized)
- ... (25 total pedagogical indicators)

**Semantic Layer (384D):** The `SentenceTransformer (all-MiniLM-L6-v2)` model encodes the problem text into a dense 384-dimensional vector. This captures the *meaning* and *domain* of the academic content, enabling the policy to generalize across subjects.

**Code Reference:** `full_pipeline_groq_2000.py`, lines 17-24 (SentenceTransformer loading), lines 151-152 (state concatenation).

---

## 4. Training Algorithm: Conservative Q-Learning (CQL)

### Why CQL? The Overestimation Problem in Offline RL
In **Offline RL**, we train from a fixed dataset (no live interaction). The danger is that the model may overestimate the value of actions it has never seen in the data. CQL prevents this by adding a **conservative penalty**:

$$L_{CQL} = \alpha \cdot \left[ \log \sum_{a} \exp(Q(s,a)) - \mathbb{E}_{a \sim D}[Q(s,a)] \right] + \frac{1}{2} \mathbb{E}_{(s,a,s') \sim D}\left[(Q - \hat{\mathcal{B}}Q)^2\right]$$

**In simple terms:**
- **First term (Conservative Penalty):** Pushes down Q-values for ALL actions, especially unseen ones.
- **Second term (Standard TD Loss):** Pulls up Q-values for actions that actually appeared in our dataset.
- **$\alpha = 4.0$:** Controls the conservatism level. Higher = safer but less creative. We tuned this via grid search.

### Training Hyperparameters

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| Total Steps | 1,000,000 | Publication-grade convergence |
| Steps/Epoch | 10,000 | Granular logging |
| Batch Size | 128 | Stable gradient estimation |
| Learning Rate | 3e-5 | Prevents catastrophic updates |
| $\alpha$ (CQL Penalty) | 4.0 | "Pedagogical Sweet Spot" — not too rigid, not too risky |
| $\gamma$ (Discount) | 0.9 | Long-term student success matters more than one good hint |

**Code Reference:** `run_full_training.py`, lines 32-49.

---

## 5. Dense Reward Shaping (Heuristics)

### The Problem With Sparse Rewards
If we only give +1 when the student solves the problem (after 10 turns), the model has no signal to learn from during the first 9 turns. This makes training extremely slow and unstable.

### Our Dense Reward Function

$$R_t = \underbrace{+1.0}_{\text{if solved}} + \underbrace{+0.2 \cdot \mathbb{I}(\text{Effort})}_{\text{student wrote formula}} + \underbrace{(-0.1) \cdot \mathbb{I}(\text{Frustration})}_{\text{student is upset}}$$

| Heuristic | Signal | Value | Why It Helps |
|:----------|:-------|:------|:-------------|
| **Effort Detection** | Student writes a formula or logical step | +0.2 | Teaches model that provoking attempts is good |
| **Frustration Penalty** | Student says "I don't know" or wants to quit | -0.1 | Teaches model to switch to "Encourage" action |
| **Goal Success** | Student arrives at correct answer | +1.0 | Terminal reward for episode success |
| **HER Relabeling** | Failed episode but sub-goal achieved | +0.5 | Recovers learning signal from failures |

**Code Reference:** `full_pipeline_groq_2000.py`, lines 171-190 (reward computation and HER logic).

---

## 6. Hindsight Experience Replay (HER)

### Concept
Most initial tutoring sessions end in failure. Discarding this data wastes valuable information. HER asks: *"Even though the student didn't solve the final problem, did they achieve any intermediate milestone?"*

### How It Works in Our Code
```
Turn 1: Student says "I have no idea"         → Reward: 0.0
Turn 2: Tutor gives a hint                    → Reward: 0.0
Turn 3: Student writes correct formula!       → Effort flag (Index 16) = 1
Turn 4: Student gets confused again           → Reward: -0.1
...
Turn 10: Student fails final answer           → Original: -1.0
                                               → HER Relabel: +0.5 (because Turn 3 was a sub-goal success)
```

**Impact:** HER increased positive gradient signals by **~450%**, enabling the model to reach 60%+ success rates with significantly less data.

**Code Reference:** `full_pipeline_groq_2000.py`, lines 181-187.

---

## 7. Multi-Persona Data Generation

### Why Diverse Student Personas Matter
If we train only on "average" students, the model fails when encountering edge cases (extremely shy or extremely overconfident students). We simulate 3 distinct personas:

| Persona | Behavior Pattern | RL Challenge | Example Response |
|:--------|:----------------|:-------------|:-----------------|
| **Shy/Struggling** | Says "Idk", needs hand-holding | Must learn to use "Encourage" | *"I'm not good at this..."* |
| **Overconfident** | Insists wrong answer is correct | Must learn to use "Verify" | *"The answer is 50, I'm 100% sure!"* |
| **Distracted** | Talks about movies, goes off-topic | Must learn to use "Guide" | *"Did you watch the new Spiderman?"* |

**Code Reference:** `full_pipeline_groq_2000.py`, lines 51-55 (persona definitions).

---

## 8. Explainable RL (XRL)

Standard RL models are "black boxes" — they choose an action but don't explain why. We implemented **Chain-of-Thought prompting** to force the tutor to output its reasoning before every hint:

```
Output Format:
[Reasoning] The student expressed frustration (Index 9 = 1). Based on the CQL policy,
            "Encourage" has the highest Q-value in this state. Switching from "Instruct"
            to reduce cognitive overload.
[Tutor Hint] You're doing amazing! Let's simplify this — what's the very first number
             in the problem?
```

This makes the system **transparent** and **auditable** — critical for educational AI deployment.

**Code Reference:** `full_pipeline_groq_2000.py`, lines 89-109 (explainable tutor logic).

---

## 9. Results

### 9.1 Cross-Domain Success Rate

| Domain | Baseline (25D) | Our Model (409D) | Improvement |
|:-------|:--------------:|:----------------:|:-----------:|
| Mathematics | 60.33% | **64.12%** | +6.3% |
| Physics | 27.48% | **62.80%** | +128% |
| History | 25.10% | **62.60%** | +149% |
| Programming | 24.50% | **61.90%** | +152% |

**Key Insight:** The baseline collapses on unseen domains (Physics: 27%), proving "Domain Blindness." Our 409D model maintains ~63% across ALL domains — a **2.3x improvement** in generalization.

### 9.2 Convergence Speed
- **With Dense Rewards:** Stable policy in ~10 epochs.
- **With Sparse Rewards (Baseline):** Requires 30+ epochs, high variance.
- **Speedup:** 3x faster convergence.

### 9.3 Action Distribution (Learned vs Default LLM)

| | Direct Answer | Encourage | Verify | Guide |
|:-|:---:|:---:|:---:|:---:|
| **Default LLM** | 90% | 5% | 3% | 2% |
| **Our RL Policy** | 20% | 25% | 30% | 25% |

The RL policy mimics expert human tutors who use diverse strategies instead of just giving answers.

---

## 10. Project Architecture & File Map

```
rl_paper/
│
├── full_pipeline_groq_2000.py      # Core: Data generation (2000 dialogues) + CQL training
├── run_full_training.py            # Standalone CQL training (1M steps)
├── run_real_free_pipeline.py       # 409D state extraction + reward shaping logic
├── problems_zeroshot.py            # Zero-shot domain problems (Biology, Chemistry, etc.)
├── generate_synthetic_dataset.py   # Synthetic dataset bootstrapping
│
├── simulator_dashboard.py          # Streamlit interactive dashboard
├── viva_demo_trace.py              # Step-by-step mathematical trace for viva
│
├── plot_results.py                 # Success rate benchmark graphs
├── plot_efficiency.py              # Convergence speed graphs
├── plot_innovation_vs_paper.py     # 409D vs 25D comparison graphs
├── generate_all_presentation_graphs.py  # All research-grade figures
│
├── ReadMe_divyansh_kumar.txt       # Team details & contributions
├── Execution-Step_divyansh_kumar.txt   # Step-by-step execution guide
├── Dataset_divyansh_kumar.txt      # Dataset documentation
├── Implemented-Ideas_divyansh_kumar.txt # Innovation details with results
├── RESEARCH_PAPER_REPORT.md        # Full research paper report
│
├── .env                            # API key (excluded from git)
├── .gitignore                      # Security & cleanup rules
└── d3rlpy_logs/                    # Training metrics & checkpoints
```

---

## 11. Environment Setup & Execution

### Prerequisites
- Python 3.8+
- Groq API Key ([console.groq.com](https://console.groq.com))

### Installation
```bash
# Option A: pip
pip install d3rlpy sentence-transformers openai streamlit python-dotenv numpy matplotlib

# Option B: conda
conda create -n rl_tutor python=3.10
conda activate rl_tutor
pip install d3rlpy sentence-transformers openai streamlit python-dotenv numpy matplotlib
```

### Configuration
```bash
# Create .env file in project root
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### End-to-End Pipeline Execution
```bash
# Step 1: Generate 2000 pedagogical dialogues
python full_pipeline_groq_2000.py

# Step 2: Train CQL model (1,000,000 steps)
python run_full_training.py

# Step 3: Generate benchmark graphs
python plot_results.py
python plot_efficiency.py
python plot_innovation_vs_paper.py

# Step 4: Launch interactive dashboard
streamlit run simulator_dashboard.py

# Step 5 (Optional): Run viva demonstration trace
python viva_demo_trace.py
```

---

## 12. Team Contributions

### Divyansh Kumar (U23AI082) — Team Leader
**Role: Core Research + Implementation + Replication**

Divyansh led the end-to-end technical execution of the project. He was responsible for replicating the Stanford baseline paper's results using open-source LLMs (Llama-3-8B via Groq), achieving the target 60.33% success rate locally. He then designed and implemented the novel 409-dimensional state representation by integrating SentenceTransformers for semantic awareness. He wrote the complete training pipeline (`full_pipeline_groq_2000.py`, `run_full_training.py`), implemented the Dense Reward Shaping heuristics, coded the Hindsight Experience Replay (HER) logic, and configured the CQL hyperparameters through systematic tuning.

### Tanu Meena (U23AI076)
**Role: Idea Replication + Benchmark Validation**

Tanu focused on the critical phase of scientific replication and evaluation. She was responsible for meticulously reproducing the experimental metrics from the reference paper to establish a robust baseline. Her work included managing the cross-domain evaluation tasks (Math, Physics, History), designing the evaluation scripts (`plot_results.py`, `plot_efficiency.py`), and ensuring that the reproduced results were statistically consistent with the original paper's reported 60.33% success rate. Her stress-testing of the baseline on unseen domains directly exposed the "Domain Blindness" flaw.

### Hanmant Jajulwar (U23AI098)
**Role: Conceptual Vision + UI/UX Development**

Hanmant provided the foundational conceptual vision for the project, identifying the key pedagogical limitations in standard LLMs that initiated this research. He proposed the RL-based tutoring approach and the multi-persona simulation strategy. He led the development of the interactive Streamlit dashboard (`simulator_dashboard.py`) and the Explainable AI (XRL) interface, translating complex RL decisions into a transparent, user-friendly demonstration. He also conducted behavioral testing across all 3 student personas to ensure robustness.

---

## 13. Future Scope

1. **Multi-Modal State (1000D+):** Add voice emotion analysis and facial expression features from webcam to capture richer student context.
2. **Continuous Action Space:** Move from 4 discrete actions to a continuous space where the model controls both *strategy* and *tone/length* of the response.
3. **Online Fine-tuning:** Deploy the model in a real LMS (Learning Management System) and perform A/B testing with actual student cohorts.
4. **Dynamic Reward Shaping:** Personalize the reward function per student profile — a shy student's "frustration" threshold should differ from a confident student's.
5. **Universal Pedagogy:** Scale to a single model that can teach any subject from primary school to graduate level.

---

## 14. Key Technical Terms (Quick Reference)

| Term | Definition |
|:-----|:-----------|
| **MDP** | Markov Decision Process — mathematical framework for sequential decisions |
| **CQL** | Conservative Q-Learning — prevents overestimation of unseen actions in offline RL |
| **HER** | Hindsight Experience Replay — relabels failed episodes using achieved sub-goals |
| **409D State** | Our innovation: 25D behavioral + 384D semantic feature fusion |
| **Dense Reward** | Intermediate rewards at every turn (vs sparse: only at episode end) |
| **XRL** | Explainable RL — model outputs reasoning before taking action |
| **Offline RL** | Learning from a fixed dataset without live interaction |
| **$\alpha$** | CQL conservatism coefficient (higher = more conservative) |
| **$\gamma$** | Discount factor (higher = values long-term outcomes more) |
| **TD Loss** | Temporal Difference error — measures prediction accuracy of Q-values |
| **D+ Pipeline** | Optimism-Guided Data Augmentation — amplifies successful trajectories |

---

## 15. References

1. Lam, N., et al. "Optimising Multi-Turn Student Outcomes with LLMs." Stanford University (2024).
2. Kumar, A., et al. "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS (2020).
3. Andrychowicz, M., et al. "Hindsight Experience Replay." NeurIPS (2017).
4. Sutton, R.S. & Barto, A.G. "Reinforcement Learning: An Introduction." MIT Press (2018).
5. Reimers, N. & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP (2019).

---

*Project developed as part of the Reinforcement Learning course curriculum. All training performed on synthetically generated data to ensure ethical compliance and student data privacy.*

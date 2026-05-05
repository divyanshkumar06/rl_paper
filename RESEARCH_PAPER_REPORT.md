# Efficient Reinforcement Learning for Optimising Multi-turn Student Outcomes with LLM Tutors

**Authors:** Divyansh Kumar, Tanu Meena, Hanmant Jajulwar  
**Institution:** Department of Artificial Intelligence  
**Date:** May 2026

---

## Abstract
Traditional Large Language Model (LLM) based tutoring systems often struggle with maintaining pedagogical consistency across multi-turn interactions and generalizing to diverse academic domains. This research presents an advanced framework utilizing **Offline Reinforcement Learning** to optimize tutoring strategies. We introduce a novel **409-Dimensional Context-Aware State Representation**, merging 25D behavioral features (affective state) with 384D semantic embeddings (cognitive state). By employing **Conservative Q-Learning (CQL)** to mitigate distribution shift and **Hindsight Experience Replay (HER)** to maximize data efficiency, our system achieves a publication-grade success rate of ~63.2% across multiple unseen domains (Mathematics, Physics, History, Programming). This demonstrates a 2.3x improvement over standard behavioral-only baselines. Our findings indicate that dense reward shaping, semantic awareness, and "optimism-guided" data augmentation are critical for creating robust, generalizable, and explainable AI tutors capable of mimicking expert human pedagogical reasoning.

---

## 1. Introduction

### 1.1 Background: The Evolution of Educational Technology
The integration of Artificial Intelligence in education has evolved through three distinct waves:
1. **Rule-Based Systems (1980s-2000s):** Fixed "if-then" logic for hint delivery.
2. **Predictive Analytics (2010s):** Using regression to predict student failure.
3. **Generative & Agentic AI (Present):** Using LLMs and Reinforcement Learning to create dynamic, human-like tutoring experiences.

In a modern classroom setting, a tutor's goal is not merely to provide the answer, but to foster "Productive Struggle"—a state where the student is challenged enough to learn but supported enough to not give up. This research addresses the transition from "Answer-Giving" machines to "Pedagogical Coaches."

### 1.2 Problem Statement: The "Domain Blindness" & "Answer-Giving" Bias
Current LLMs suffer from two primary flaws in education:
1. **Answer-Giving Bias:** Pre-trained on helpfulness as a general assistant, LLMs tend to give the solution directly when asked, which bypasses the student's cognitive processing and hinders long-term retention.
2. **Domain Blindness:** Recent research (Stanford baseline) successfully used 25-dimensional behavioral vectors to train RL tutors. However, these models were "domain-blind"—their effectiveness was limited to the specific subjects they were trained on (e.g., Mathematics). When presented with a Physics or History problem, the lack of semantic context caused the tutoring policy to collapse, as the model could not differentiate between the logical structures of different subjects.

### 1.3 Motivation and Research Questions
The motivation for this work is to bridge the gap between "Generative AI" (LLMs) and "Agentic Strategic Decision Making" (Reinforcement Learning). 
- **RQ1:** Can we represent the academic context and student behavior in a unified vector space?
- **RQ2:** Does Offline RL provide a safe and stable way to optimize tutoring strategies without live student risk?
- **RQ3:** Can a single tutoring policy generalize across diverse subjects like History and Physics?

---

## 2. Mathematical Framework: MDP for Tutoring

### 2.1 The Markov Decision Process (MDP)
We formalize tutoring as a finite MDP defined by the tuple $(S, A, P, R, \gamma)$:
- **State Space ($S$):** The 409-dimensional vector representing the current turn's context.
- **Action Space ($A$):** Four discrete pedagogical strategies.
- **Transition Probability ($P$):** The probability $P(s_{t+1} | s_t, a_t)$, which is non-deterministic as it depends on the student's response.
- **Reward Function ($R$):** The dense heuristic mapping $R: S \times A \to \mathbb{R}$.
- **Discount Factor ($\gamma$):** Set to 0.9 to prioritize strategies that lead to successful long-term outcomes over immediate gratification.

### 2.2 The Bellman Optimality Equation
The goal of our CQL agent is to find the optimal policy $\pi^*$ that maximizes the expected cumulative reward:
$$ Q^*(s, a) = \mathbb{E} [R_t + \gamma \max_{a'} Q^*(s_{t+1}, a') | s_t=s, a_t=a] $$

---

## 3. The 409-Dimensional Architecture (Core Innovation)

### 3.1 Pseudo-code: State Extraction Logic
```python
def extract_context_aware_state(dialogue_history, problem_text):
    # 1. Behavioral Layer (Affective)
    # LLM classifies 25 features: Frustration, Confusion, etc.
    behavioral_25d = llm_classifier(dialogue_history)
    
    # 2. Semantic Layer (Cognitive)
    # Transformer encodes the actual subject matter
    semantic_384d = transformer_model.encode(problem_text)
    
    # 3. Concatenation (Feature Fusion)
    state_409d = np.concatenate([behavioral_25d, semantic_384d])
    return state_409d
```

### 3.2 Dimensionality Rationale
The 25 behavioral dimensions capture "how" the student is saying it (Affective state). The 384 semantic dimensions capture "what" they are talking about (Cognitive state). This fusion is critical because a "Struggling" student in Math needs a procedural hint, whereas a "Struggling" student in History might need a contextual clue or a date reference.

---

## 4. Multi-Persona Case Studies

### 4.1 Persona A: The "Shy/Struggling" Student
- **Profile:** Low confidence, frequent "I don't know" responses.
- **RL Strategy Learned:** The policy shifts toward the "Encourage" action.
- **Dialogue Example:** 
  - *Tutor:* "Let's try Step 1 together."
  - *Student:* "I'm not good at this, I don't know."
  - *Model Action:* [Encourage] -> "You're doing great! Even starting is a big step. What do you think is the first number we should look at?"

### 4.2 Persona B: The "Overconfident" Student
- **Profile:** High confidence, frequent calculation errors, bypasses steps.
- **RL Strategy Learned:** The policy shifts toward "Verify Sub-step."
- **Dialogue Example:**
  - *Student:* "The answer is 50, I'm 100% sure!"
  - *Model Action:* [Verify] -> "That's a bold answer! Before we finalize, can you show me how you calculated the last multiplication?"

---

## 5. System Implementation Details

### 5.1 Hindsight Experience Replay (HER) Mechanism
HER is crucial for educational data. Since tutoring paths are long, failures are common.
- **Original Path:** Turn 1 -> Turn 10 (Failure, Reward: -1)
- **HER Path:** At Turn 5, the student correctly identified the formula. We create a "Virtual Goal" where Turn 5 is the success point. We relabel this transition in the buffer as a success (+1.0), teaching the model that the *path* taken to Turn 5 was correct, even if the final answer was wrong.

### 5.2 Training Stability with CQL
One of the major challenges in Offline RL is "Action Distribution Shift." If the model chooses an action the LLM didn't take in the dataset, it has no ground truth. CQL solves this by penalizing the Q-values of actions not present in the dataset:
$$ L(Q) = \alpha \cdot \text{LogSumExp}(Q) - \mathbb{E}[Q_{dataset}] $$
This ensures the AI tutor stays "humble" and sticks to pedagogical strategies validated by the training data.

---

## 6. Detailed Results & Statistical Significance

### 6.1 Success Rate Across Domains (Normalized)
The following success rates were achieved after 1,000,000 steps of training on the augmented D+ dataset:
1. **Mathematics (GSM8K):** 64.12% (Baseline 25D: 60.33%)
2. **Physics (Kinematics):** 62.80% (Baseline 25D: 27.48%) - **Crucial Proof of Semantic Value**
3. **US History:** 62.60% (Baseline 25D: 25.10%)
4. **Python Programming:** 61.90% (Baseline 25D: 24.50%)

### 6.2 Convergence Stability
In standard Q-learning, the TD Error often spikes. Our CQL implementation, combined with Dense Reward Shaping, showed a smooth exponential decay of TD error, reaching a steady state of <0.05 within the first 15 epochs.

---

## 6. Implementation & Technical Stack
- **Languages:** Python 3.10
- **RL Framework:** `d3rlpy` (Discrete CQL)
- **NLP:** `sentence-transformers`, `openai` (Groq API), `llama-3-8b`
- **Dashboard:** `streamlit` (Explainable XRL Dashboard)
- **Data:** `pickle` for serialized transition buffers.

---

## 7. Comprehensive Methodology: The Four Phases of Development

### Phase 1: Environment Simulation & Persona Design
The first challenge was creating a non-deterministic environment that reflects a real classroom. Unlike standard RL environments (e.g., Atari games), tutoring responses are highly subjective.
- **Action:** We engineered Llama-3-8B with system prompts that enforce strict persona adherence.
- **Challenge:** LLMs tend to be "too smart" or "too helpful." 
- **Solution:** We added "Mistake Injection" logic where the student persona is forced to make a logical error (e.g., miscalculation) in Turn 2 to test the tutor's corrective behavior.

### Phase 2: The 409D Feature Fusion
This phase involved the pipeline integration of `SentenceTransformers`.
- **Logic:** The semantic vector (384D) acts as a "Memory Anchor," ensuring that the policy remains consistent with the specific problem's logical tree.
- **Verification:** We used T-SNE plots to verify that Math problems clustered separately from History problems in the state space.

### Phase 3: Offline Optimization (CQL)
We utilized the `d3rlpy` library to perform offline optimization.
- **Hyperparameter Tuning:** We conducted a grid search for the $\alpha$ coefficient. High $\alpha$ (e.g., 10) made the tutor too repetitive; low $\alpha$ (e.g., 1) caused the tutor to suggest random topics. We settled on **$\alpha=4.0$** as the "Pedagogical Sweet Spot."

### Phase 4: Validation and Benchmarking
We used "Leave-One-Domain-Out" cross-validation to test the 409D innovation's robustness. This is where we proved the 2.3x improvement in generalization.

---

## 8. Ethical Considerations & AI Safety
In educational AI, safety is paramount.
1. **Bias Mitigation:** We screened the semantic embeddings to ensure no subject-specific biases were learned.
2. **Pedagogical Integrity:** The system is hard-coded to never give the final answer unless 10 turns have passed and the student is still stuck (The "Safety Valve").
3. **Data Privacy:** All training was performed on synthetic data to avoid the risks associated with student PII (Personally Identifiable Information).

---

## 9. Limitations and Mitigation Strategies
| Limitation | Mitigation |
| :--- | :--- |
| **Fixed Action Space** | We implemented "Reasoning Blocks" to add nuance to the fixed actions. |
| **API Latency** | Used Groq's high-speed inference to ensure data generation didn't take weeks. |
| **Semantic Drift** | Periodic recalibration of the SentenceTransformer embeddings. |

---

## 10. Conclusion & Strategic Roadmap
We have successfully demonstrated that Offline Reinforcement Learning, when coupled with a rich, context-aware state representation, can produce AI tutors that are significantly more robust than vanilla LLMs. 

**Strategic Roadmap:**
1. **Short Term:** Integration of multi-persona support for collaborative learning groups.
2. **Medium Term:** Deploying the model as a backend for an open-source LMS.
3. **Long Term:** Exploring "Universal Pedagogy" where a single model can teach any subject from Kindergarten to PhD level.

---

## 11. Appendix: System Requirements
- **Inference:** Minimum 8GB VRAM for local Transformer processing.
- **Storage:** ~500MB for the serialized offline dataset.
- **Connectivity:** Required for LLM API calls during the data generation phase.

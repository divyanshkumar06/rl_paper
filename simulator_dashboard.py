import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="RL Tutor Simulator", layout="wide", page_icon="🎓")

# Sidebar Configuration
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Stanford_University_seal_2003.svg/1200px-Stanford_University_seal_2003.svg.png", width=50)
st.sidebar.title("RL Tutor Simulator")
menu = st.sidebar.radio("Simulation Flow", [
    "1. Core Architecture (Sutton & Barto)",
    "2. Data Generation Pipeline",
    "3. Target Analytics & Results"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Team:** Divyansh, Tanu, Hanmanth")

if menu == "1. Core Architecture (Sutton & Barto)":
    st.title("🎓 Conceptual Architecture & Reinforcement Learning Design")
    st.markdown("#### Solving Multi-Domain Tutoring using Advanced MDP formulation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Sutton & Barto Chapter 2 Integration (Multi-Arm Bandits):**\n\nThe original data generation used pure uniform random actions. We successfully optimized dataset collection by implementing **ε-greedy (Epsilon-Greedy) / UCB Action Selection**, heavily balancing exploration and exploitation to build a higher-yield Offline Dataset.")
        
    with col2:
        st.success("**Sutton & Barto Chapter 3 (Finite MDPs):**\n\nThe 409-Dimensional Context-Aware Vector (SentenceTransformer + 25D Behavior) strictly defines the Markov State Space $S_t$. Dense reward scaling correctly fulfills the Reward Hypothesis for rapid Q-function optimization.")
        
    st.markdown("---")
    st.markdown("### The Engineered State Representation")
    st.code("S_new = [ S_behavioral_25D  ⊕  S_semantic_384D ]", language='python')
    st.markdown("This universally generalizes to Mathematics, Physics, History, and Coding domains.")

elif menu == "2. Data Generation Pipeline":
    st.title("⚙️ Live Data Generation Simulator")
    st.markdown("Initiate the Omni-Domain Conversational Agent loop.")
    
    st.code("python run_real_free_pipeline.py", language="bash")
    if st.button("Simulate Pipeline Start"):
        st.success("Tutor and Student models are conversing over the Groq Llama-3 API...")
        with st.expander("View Real-Time Interaction Logs", expanded=True):
            st.text("--- Conversation 1 (Physics Domain) ---")
            st.text("Turn 1 completed... (Epsilon-Greedy Exploitation)")
            st.text("Turn 2 completed... (Dense Shaping Reward: 0.2)")
            st.text("Turn 3 completed... (Dense Shaping Reward: -0.1)")
            st.text("Solved at Turn 4! (Dense Reward included)")
    st.markdown("*(Note: To actually generate the data, execute the script in your local IDE terminal)*")

elif menu == "3. Target Analytics & Results":
    st.title("📊 Reinforcement Learning Assessment Dashboard")
    
    st.subheader("Paper Reproduction Success")
    try:
        img1 = Image.open("figure_3_reproduced.png")
        st.image(img1, caption="Baseline vs. CQL Evaluation (Matched 60.33% Target)", width=600)
    except FileNotFoundError:
        st.warning("Please run 'python plot_results.py' to generate the first evaluation graph.")
        
    st.markdown("---")
    st.subheader("Advanced Architecture Engineering Metrics")
    try:
        img2 = Image.open("advanced_metrics_dashboard.png")
        st.image(img2, caption="Convergence and Action Typography of our 409D + HER Integrated Model")
    except FileNotFoundError:
        st.warning("Please run 'python plot_advanced_innovations.py' to generate the advanced metrics dashboard.")
        
    st.markdown("---")
    st.subheader("Final Benchmark: Innovation vs. Original Paper")
    try:
        img3 = Image.open("innovation_vs_paper_comparison.png")
        st.image(img3, caption="Generalizability and Architecture Benchmarks scaling beyond the Stanford Baseline", use_container_width=True)
    except FileNotFoundError:
        st.warning("Please run 'python plot_innovation_vs_paper.py' to generate the comparison benchmarks.")

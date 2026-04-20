# Zero-Shot Domain Transfer Problems
# These problems are used to test if the model trained on Math/Physics
# can generalize its pedagogical strategy to New Subjects.

ZERO_SHOT_PROBLEMS = [
    {
        "domain": "Chemistry",
        "problem": "Balance the following equation: H2 + O2 -> H2O. Answer: 2H2 + O2 -> 2H2O.",
        "mistake": "The student forgets that Oxygen is diatomic and tries to balance it as H + O -> HO."
    },
    {
        "domain": "Biology",
        "problem": "What are the four bases of DNA? Answer: Adenine, Thymine, Cytosine, Guanine.",
        "mistake": "The student replaces Thymine with Uracil (which is for RNA)."
    },
    {
        "domain": "Computer Science",
        "problem": "What is the time complexity of searching for an element in a balanced Binary Search Tree? Answer: O(log n).",
        "mistake": "The student says O(n) because they think they might have to check every node like a list."
    },
    {
        "domain": "Psychology",
        "problem": "Who developed the theory of Classical Conditioning? Answer: Ivan Pavlov.",
        "mistake": "The student confuses Pavlov with B.F. Skinner (Operant Conditioning)."
    }
]

# 🔍 Uncertainty in Large Language Models (LLMs)

This repository contains the research code, experiments, and documentation for my Master's thesis, which explores **uncertainty estimation in Large Language Models (LLMs)**. The focus is on identifying, quantifying, and interpreting different forms of uncertainty—particularly **semantic uncertainty**, and using metrics such as **Semantic Answer Rating (SAR)** and **Negative Log-Likelihood (NLL)** to evaluate model confidence and reliability.

---

## 📚 Thesis Overview

Modern LLMs are powerful but inherently uncertain in their outputs. Understanding and quantifying this uncertainty is critical for safe and trustworthy AI applications. This thesis aims to:

- Investigate various sources and types of uncertainty in LLM outputs
- Analyze **semantic uncertainty**: when the model output is syntactically correct but semantically vague, ambiguous, or misleading
- Use and evaluate uncertainty estimation metrics like:
  - **Negative Log-Likelihood (NLL)**
  - **Entropy-based methods**
  - **Semantic Entropy**
  - **Shifting Attention to Relevance (SAR)**
- Explore how uncertainty correlates with downstream task performance and human-perceived quality
- Propose or evaluate methods to *reduce* or *calibrate* uncertainty in LLM predictions

---

## ⚙️ Features

- 📊 Code for computing uncertainty metrics across LLM responses
- 🧪 Scripts for running experiments on benchmark datasets (e.g., QA, summarization, etc.)
- 🤖 Integration with HuggingFace Transformers for model inference
- 📈 Visualization tools for interpreting and comparing uncertainty scores
- 📝 Dataset support for semantic evaluation

---

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch / HuggingFace Transformers
- NumPy, SciPy, Scikit-learn
- Matplotlib / Seaborn for visualizations
- Jupyter Notebooks for experimentation and reporting

---

## 🧪 Example Metrics

- **Negative Log-Likelihood (NLL)**  
  Measures the likelihood of the model output under its predicted distribution.

- **Semantic Uncertainty**  
  A proposed or adopted metric to score semantic quality and alignment with expected meaning.

- **Entropy & Token-wise Variance**  
  Estimates uncertainty from the output distribution of tokens.
  
- **Shifting Attention to Relevance (SAR)**  
  A metric designed to evaluate the **semantic alignment and focus** of LLM responses. SAR quantifies how much the model’s attention shifts toward **relevant** content in relation to the prompt or task, capturing deeper semantic uncertainty.



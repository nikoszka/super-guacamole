# Token Visualization App Guide

## Overview

The **Token Visualization App** is an interactive Streamlit application that lets you explore how language models generate text at the token level. You can visualize probabilities, uncertainties, and see how each token contributes to the final answer.

## ✨ New Features

### 1. **Probability Visualization** 
- View token-level probabilities (model confidence) instead of just NLL
- Green color intensity shows higher confidence
- Toggle between Probability and NLL views

### 2. **Question & Answer Context**
- See the original question for each example
- View the correct answer(s) from the dataset
- Compare model response with correct answer

### 3. **Short & Long Answer Support**
- Switch between short answer datasets (brief responses)
- Or long answer datasets (detailed explanations)
- Pre-configured paths for both types

## 🚀 Quick Start

### Launch the App

```bash
# Option 1: Using the launcher script (recommended)
python run_token_viz_app.py

# Option 2: Direct streamlit command
streamlit run src/analysis/token_visualization_app.py

# Option 3: Custom port
python run_token_viz_app.py --port 8502
```

The app will open in your browser at `http://localhost:8501`

## 📊 Using the App

### Sidebar Controls

1. **Answer Type**: Choose between "Long Answers" or "Short Answers"
   - Long: Detailed explanations with more tokens
   - Short: Brief, concise answers

2. **Pickle File Path**: Path to the validation pickle file
   - Default paths are pre-filled based on answer type
   - You can paste a custom path to any validation_generations.pkl file

3. **Example Slider**: Navigate through examples (1 to N)

4. **Score Type**: Choose what to visualize
   - **Probability**: Green = high confidence
   - **NLL**: Red = high uncertainty

### Main Display

#### Question & Answer Section
- **Question**: The original question from the dataset
- **Correct Answer**: The ground truth answer(s)

#### Model Response Section
- **Accuracy**: ✅ or ❌ indicator with score
- **Response Text**: What the model generated

#### Token Visualization
- **Probability Mode** (Green):
  - Darker green = higher probability (more confident)
  - Lighter green = lower probability (less confident)
  - Each token shows its probability value below

- **NLL Mode** (Red):
  - Darker red = higher NLL (more uncertain)
  - Lighter red = lower NLL (more certain)
  - Each token shows its NLL value below

#### Statistics Panel
- Mean, Min, Max, and Standard Deviation
- Helps identify patterns in model behavior

## 📂 Data Sources

### Default Pickle Paths

**Long Answers (Judge-Corrected):**
```
src/boldis/uncertainty/wandb/run-20251121_092732-wiboofpr/files/validation_generations_judge_corrected.pkl
```

**Short Answers:**
```
src/boldis/uncertainty/wandb/run-20251121_011028-sqykmrn7/files/validation_generations.pkl
```

### Custom Pickle Files

You can use any validation pickle file that has the following structure:

```python
{
    'example_id': {
        'question': str,
        'reference': {
            'answers': {'text': List[str]}
        },
        'most_likely_answer': {
            'response': str,
            'tokens': List[str],
            'token_log_likelihoods': List[float],
            'accuracy': float
        }
    }
}
```

## 🎨 Visualization Features

### Color Schemes

- **Green (Probability)**: 
  - rgba(0, 200, 0, alpha)
  - Higher alpha = higher probability
  
- **Red (NLL)**:
  - rgba(255, 0, 0, alpha)
  - Higher alpha = higher uncertainty

### Interactive Elements

- Hover over tokens to see details (browser dependent)
- Tokens wrap naturally for long sequences
- Responsive layout adapts to window size

## 💡 Use Cases

### 1. Analyzing Model Confidence
- Identify which tokens the model is uncertain about
- Find patterns in high/low confidence regions
- Compare confidence across different answer types

### 2. Debugging Wrong Answers
- See where the model went wrong
- Check if low probability tokens correlate with errors
- Compare correct vs incorrect predictions

### 3. Understanding Answer Quality
- Long answers: Are detailed explanations confident?
- Short answers: Does brevity mean higher confidence?
- Identify "hallucination zones" (high confidence, wrong answer)

### 4. Comparative Analysis
- Compare short vs long answer confidence patterns
- Analyze first-token probabilities
- Study how confidence evolves through the sequence

## 🔧 Troubleshooting

### File Not Found Error
```
File not found: <path>
```
**Solution**: 
- Check that the pickle file exists at the specified path
- Use absolute or relative paths correctly
- Ensure you're running from the project root directory

### No Valid Examples
```
No valid examples found in pickle file.
```
**Solution**:
- Pickle file must contain `most_likely_answer` with `tokens` and `token_log_likelihoods`
- Regenerate answers with token storage enabled
- Check pickle structure with: `python -c "import pickle; print(pickle.load(open('path.pkl', 'rb')).keys())"`

### Slow Loading
**Solution**:
- Large pickle files are cached automatically
- First load may take time, subsequent loads are fast
- Consider using a subset of examples for exploration

## 📝 Tips & Best Practices

1. **Start with Long Answers**: More tokens = more patterns to observe
2. **Compare Probability & NLL**: Switch back and forth to understand the relationship
3. **Look for Extremes**: Very low probabilities or very high NLLs indicate interesting cases
4. **Check Question Context**: Understanding the question helps interpret the visualization
5. **Use Statistics**: Mean and std dev give you a quick overview before diving into tokens

## 🔗 Related Tools

- **Phase 1.5 Analysis**: Token-level NLL analysis scripts
- **Phase 1.6 Analysis**: Prefix NLL curves
- **Jupyter Notebooks**: `src/analysis_notebooks/` for deeper analysis

## 🆘 Getting Help

If you encounter issues:
1. Check this guide first
2. Review `ANALYSIS_README.md` for context
3. Inspect the pickle file structure
4. Check console output for error messages

## 📚 Further Reading

- `PHASE1_6_EXPLANATION.md`: Understanding NLL metrics
- `ANALYSIS_README.md`: Overview of all analysis tools
- `CODE_DOCUMENTATION.md`: Technical implementation details



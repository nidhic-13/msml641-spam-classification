# Spam Classification Project

This project implements spam classification using both a traditional NLP
(Word2Vec + Logistic Regression) and a transformer-based BERT model. The workflow includes data cleaning,
preprocessing, feature extraction, model training, and evaluation.

---
# Running the notebooks

## Environment Setup

The project is implemented in **Python 3.12** and relies on the following libraries.

## Libraries Used

- **pandas** – Data manipulation and preprocessing
- **numpy** – Numerical computations
- **tensorflow** – Deep learning framework
- **tensorflow-hub** – Pretrained model utilities
- **tensorflow-text** – Text preprocessing ops for TensorFlow
- **keras-nlp** – Pretrained BERT models and tokenizers
- **scikit-learn** – Logistic regression, evaluation metrics, and dataset splitting
- **gensim** – Word2Vec embedding training and inference
- **matplotlib / seaborn** – Visualization and analysis

### Required Libraries and Versions

| Library         | Version      |
|-----------------|--------------|
| pandas          | &ge; 2.2.2   |
| numpy           | &ge; 2.0.2   |
| tensorflow      | &ge; 2.19.0  |
| tensorflow-hub  | &ge; 0.16.1  |
| tensorflow-text | &ge; 2.19.0  |
| keras-nlp       | &ge; 0.21.1  |
| scikit-learn    | &ge; 1.6.1   |
| gensim          | &ge; 4.4.0   |
| matplotlib      | &ge; 3.10.0  |
| seaborn         | &ge; 0.13.2  |

---

## Installation

Install all required dependencies using:

```bash
pip install pandas numpy tensorflow tensorflow-hub tensorflow-text keras-nlp scikit-learn matplotlib seaborn

## Running the streamlit application
- Navigate to the streamlit directory
- Install the required dependencies from the requirements.txt located inside
- Run 'streamlit run app.py'
- The app will launch in your browser, paste your email contents into the textbox and press classify
# Twitter Sentiment Analysis Using Classical & Neural Models

## Introduction
This project studies sentiment classification on Twitter using the Kaggle "Twitter Sentiment Dataset" (Saurabh Shahane, 2021).
The dataset contains cleaned tweets (`clean_text`) and sentiment labels in `category` with values -1 (negative), 0 (neutral), +1 (positive).
The goal is to compare classical machine learning models against a text-oriented neural model (CNN–LSTM hybrid) and identify which approach
is best for multiclass sentiment classification in terms of accuracy and robust F1 (macro).


## 🎯 Research Questions Addressed
1. Which model achieves the best overall and per-class performance for predicting sentiment?
2. How do TF-IDF and Word2Vec features compare when used with classical models?
3. Does a CNN–LSTM hybrid outperform classical ensembles (Voting Classifier) on this dataset?

## Dataset
- Source: Kaggle — Twitter Sentiment Dataset (Saurabh Shahane, 2021).
- Columns: `clean_text` (string), `category` (int: -1, 0, 1).
- Size: (162980, 2)

## 📁 Project Structure
- `twitter_sentiment_main.py` - Main pipeline script that implements the entire analysis
- `Twitter_Data.csv` - Dataset with cleaned tweets and sentiment labels
- `Twitter_Sentiment_Analysis.md` - Project specification document
- `requirements.txt` - Python dependencies
- `logs/` - Directory to store log files from runs

## Methodology
1. EDA & preprocessing: class distribution, text length, missing data handling, tokenization, stopword removal.
2. Feature pipelines:
    - TF-IDF vectorizer (for baseline classical models).
    - Word2Vec embeddings (gensim): average tweet vectors for classical models.
    - Keras Tokenizer + padding for NN (embedding layer + CNN–LSTM).
3. Models:
    - Baselines: Decision Tree, KNN, Logistic Regression.
    - Ensemble: Voting Classifier (hard or soft) combining best performing classical models.
    - Neural: CNN–LSTM hybrid (Embedding → Conv1D → MaxPool → LSTM → Dense).
4. Hyperparameter tuning: GridSearchCV for classical models; manual / Keras Tuner for NN if time permits.
5. Evaluation: accuracy, precision, recall, F1 (macro & per class), confusion matrix, ROC-AUC (one-vs-rest).


## 🚀 How to Run
###  Install Dependencies
```bash
#createe a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Prepare variables (if needed)
```bash
# Set variables after importing in Twitter_Sentiment_main.py
DATA_PATH = 'twitter_data.csv' # Path to dataset
SAMPLE_SIZE = 25000 # Set to an integer for sampling, or None for full dataset
```
### 3. Run the Analysis Pipeline
```bash
**Basic usage (assuming Twitter_Data.csv is in the same directory):**
```bash
python Twitter_Sentiment_main.py
```
**To save logs to a file in /logs:**
```bash
# save log in a file 10000_run.log
python Twitter_Sentiment_main.py > logs/10000_run.log 2>&1
```

## 📊 What the Pipeline Does
1. **Exploratory Data Analysis (EDA)**
    - Analyzes sentiment distribution
    - Examines text length statistics
    - Creates visualizations

2. **Data Preprocessing**
    - Cleans text (removes URLs, mentions, hashtags)
    - Removes stopwords
    - Tokenizes text
    - Sample the dataset into a more manageable size for testing (optional)

3. **Feature Engineering**
    - **TF-IDF Features**: Creates term frequency-inverse document frequency vectors
    - **Word2Vec Features**: Generates word embeddings and averages them per tweet
    - **Neural Network Features**: Uses Keras tokenizer and padding for sequence data

4. **Model Training**
    - **Classical Models**: Decision Tree, KNN, Logistic Regression (with GridSearchCV)
    - **Ensemble**: Voting Classifier combining best models
    - **Neural Network**: CNN-LSTM hybrid architecture

5. **Evaluation**
    - Accuracy, Precision, Recall, F1-Score (macro and per-class)
    - Confusion matrices
    - ROC curves
    - Model comparison table

## 📈 Output Files
After running the pipeline, you'll get:

- **Data Files**:
    - `model_comparison.csv` - Table comparing all models


## 📝 Notes
- The script automatically downloads required NLTK data (punkt tokenizer and stopwords)
- Training time varies based on dataset size and hardware (expect 10-30 minutes for full pipeline)
- Neural network training uses early stopping to prevent overfitting


## 🔧 Troubleshooting

If you encounter memory issues with large datasets:
- Reduce `max_features` in TF-IDF vectorizer
- Decrease `batch_size` in neural network training
- Use a smaller subset of data for initial testing

## 📚 Citation

Dataset source: HUSSEIN, SHERIF (2021), "Twitter Sentiments Dataset", Mendeley Data, V1, doi:10.17632/z9zw7nt5h2.1

## 🎯 Answers to Research Questions Addressed

1. Which model achieves the best overall and per-class performance for predicting sentiment?
   The CNN-LSTM Neural Network is the clear winner, achieving the highest overall performance.

The F1-Macro score of *0.8663* is exceptional for a three-class classification task. Since F1-Macro is the unweighted average of
the F1-scores for the Negative, Neutral, and Positive classes, achieving such a high score strongly indicates that the CNN-LSTM model
has excellent, balanced performance across all three sentiment categories.

2. How do TF-IDF and Word2Vec features compare when used with classical models?

TF-IDF features significantly outperform Word2Vec features when used with the strongest classical models (Logistic Regression, Decision Tree, and Voting Classifier).

TF-IDF Dominance: TF-IDF effectively captures the importance of specific, key unigrams and bigrams (e.g., "awful," "not great") that are highly discriminative of sentiment.
This approach works better for these models than relying on continuous semantic meaning.

Word2Vec Weakness: Even with 25,000 samples, the corpus is still not large enough to train generalized, high-quality Word2Vec embeddings that are competitive with the large-scale vocabulary of Twitter.
The *100*-dimensional averaged vectors are likely too lossy and generic for accurate classification.

3. Does a CNN–LSTM hybrid outperform classical ensembles (Voting Classifier) on this dataset?

Yes, the CNN-LSTM hybrid decisively outperforms the classical ensemble (Voting Classifier).
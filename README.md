# Automated-Question-Tagging-System
ML-powered multi-label classifier that auto-generates programming tags for technical questions. Processes 1.2M+ Stack Overflow posts using NLP (TF-IDF, lemmatization) and 6 classifiers (best: SGDClassifier, 50% F1). Built with scikit-learn, NLTK, pandas for intelligent content categorization.

A machine learning pipeline for training and evaluating multi-label classification models to automatically tag Stack Overflow questions.
🎯 Overview
This project focuses on training and evaluating various machine learning models for automatic question tagging. It implements a complete NLP pipeline to process Stack Overflow data and compares 6 different classification algorithms to identify the best approach for multi-label tag prediction.
⚠️ Scope
This implementation covers:

✅ Data preprocessing and cleaning
✅ Feature extraction using TF-IDF
✅ Model training with multiple algorithms
✅ Performance evaluation and comparison

Not included:

❌ Production deployment code
❌ Prediction API/interface
❌ Model serialization/saving
❌ Real-time inference pipeline

🚀 Quick Start
Prerequisites
bashPython 3.7+
Jupyter Notebook
8GB+ RAM (for processing 1M+ questions)
Installation

Clone the repository

bashgit clone https://github.com/Tejas1712/automatic-question-tagging.git
cd automatic-question-tagging

Download Data :- https://drive.google.com/drive/folders/15_7lN5t6IZu76_4Y1pSVo4ecVnGHKZ-C?usp=drive_link


Install dependencies

bashpip install pandas numpy scikit-learn nltk beautifulsoup4 matplotlib seaborn scipy

Download NLTK resources

pythonimport nltk
nltk.download('all')  # Or specific: 'punkt', 'stopwords', 'wordnet'
Dataset Setup

Download Stack Overflow dataset:

Questions.csv - Contains Id, Title, Body, Score, etc.
Tags.csv - Contains Id, Tag pairs


Update paths in notebook:

pythonquestion = pd.read_csv("path/to/Questions.csv", encoding="ISO-8859-1")
tags = pd.read_csv("path/to/Tags.csv", encoding="ISO-8859-1")
📊 Training Pipeline
1. Data Preparation
python# Load and merge datasets
questions (1,264,216 rows) + tags (3,750,994 rows)
                    ↓
# Filter quality posts (Score > 5)
72,950 high-quality questions
                    ↓
# Select top 100 most frequent tags
Final training dataset: 63,167 questions
2. Text Preprocessing Steps
StepFunctionPurposeHTML RemovalBeautifulSoupExtract text from HTML tagsText Cleaningclean_text()Normalize contractions, remove special charsPunctuation Removalremove_punctuation()Clean while preserving programming termsLemmatizationlemitizeWords()Reduce words to base formStop WordsstopWordsRemove()Remove common English words
3. Feature Engineering
python# TF-IDF Vectorization
Title Features: TfidfVectorizer() → X1 (sparse matrix)
Body Features: TfidfVectorizer() → X2 (sparse matrix)
Combined: hstack([X1, X2]) → X (final features)

# Label Encoding
MultiLabelBinarizer() → Binary matrix (100 columns for 100 tags)
4. Model Training
python# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Training: 44,216 samples
# Testing: 18,951 samples

# Models Trained
OneVsRestClassifier(estimator) for each:
- SGDClassifier
- LogisticRegression
- MultinomialNB
- LinearSVC
- Perceptron
- PassiveAggressiveClassifier
📈 Evaluation Results
Performance Metrics
ClassifierF1 ScoreJaccard ScoreHamming LossSGDClassifier0.5045.87%0.99%Logistic Regression0.4844.23%1.02%LinearSVC0.4844.15%1.01%Passive-Aggressive0.4844.05%1.02%Perceptron0.4440.12%1.15%MultinomialNB0.4338.95%1.20%
Evaluation Function
pythondef print_score(y_pred, clf):
    # Jaccard Score: Intersection over Union for multi-label
    jacard = np.minimum(y_test, y_pred).sum(axis=1) / 
             np.maximum(y_test, y_pred).sum(axis=1)
    
    # Hamming Loss: Fraction of wrong labels
    hamming = hamming_loss(y_pred, y_test)
    
    # F1 Score: Harmonic mean of precision and recall
    f1 = metrics.classification_report(y_test, y_pred)
    
    return scores
📁 Notebook Structure
python# Section 1: Imports and Setup
- Library imports
- Warning suppression
- Style configuration

# Section 2: Data Loading
- Read Questions.csv (1.26M rows)
- Read Tags.csv (3.75M rows)
- Data exploration

# Section 3: Data Preprocessing
- Merge datasets
- Filter by score
- Select top 100 tags
- Remove nulls

# Section 4: Text Preprocessing
- HTML to text conversion
- Text cleaning pipeline
- Lemmatization
- Stop words removal

# Section 5: Feature Extraction
- TF-IDF for titles
- TF-IDF for body
- Feature combination

# Section 6: Model Training
- Train-test split
- Train 6 classifiers
- Store results

# Section 7: Evaluation
- Calculate metrics
- Compare models
- Visualize results
📊 Key Statistics
MetricValueOriginal Questions1,264,216Filtered Questions (Score>5)72,950Final Training Set63,167Feature Dimensions379,310Number of Tags100Training Samples44,216Testing Samples18,951
🔍 Important Findings

Best Performer: SGDClassifier with 50% F1 score
Feature Impact: Combining title + body improves performance
Data Quality: Filtering by score (>5) crucial for quality
Tag Distribution: Focus on top 100 tags balances coverage vs. complexity

🛠️ Customization Options
Modify Tag Count
python# Change from top 100 to different number
frequencies_words = keywords.most_common(50)  # or 200, 500, etc.
Adjust Quality Threshold
python# Change score filter
new_df = df[df['Score'] > 10]  # Stricter quality filter
Change Train-Test Split
python# Adjust split ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
📦 Dependencies
pythonpandas==1.5.3          # Data manipulation
numpy==1.24.3          # Numerical operations
scikit-learn==1.2.2    # ML algorithms
nltk==3.8.1            # NLP preprocessing
beautifulsoup4==4.12.2 # HTML parsing
matplotlib==3.7.1      # Visualization
seaborn==0.12.2        # Statistical plots
scipy==1.10.1          # Sparse matrix operations
⚡ Performance Considerations

Memory Usage: ~4-6 GB during training
Training Time: ~15-30 minutes for all models
Sparse Matrices: Essential for handling 379K features
Batch Processing: Consider chunking for larger datasets

🔬 Future Research Directions

Experiment with different vectorization techniques (Word2Vec, Doc2Vec)
Try ensemble methods combining multiple classifiers
Test with different numbers of top tags (200, 500)
Implement cross-validation for robust evaluation
Analyze per-tag performance metrics
Study impact of different preprocessing steps

📝 Notes

This is a training/evaluation pipeline, not production-ready code
Models are not saved - add pickle/joblib for persistence
No hyperparameter tuning implemented - consider GridSearchCV
Evaluation on same tag set as training - consider zero-shot evaluation

🤝 Contributing
Feel free to experiment with:

Different preprocessing techniques
Additional classification algorithms
Alternative feature extraction methods
Hyperparameter optimization

🙏 Acknowledgments

Stack Overflow for the dataset
scikit-learn for ML implementations
NLTK for NLP tools

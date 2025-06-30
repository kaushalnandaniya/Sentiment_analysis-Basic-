# Sentiment Analysis with Logistic Regression

A machine learning project that performs sentiment analysis on Twitter data using Logistic Regression and TF-IDF vectorization.

## 📋 Project Overview

This project implements a sentiment analysis classifier that can determine whether a given text expresses positive or negative sentiment. The model is trained on the Sentiment140 dataset, which contains 1.6 million tweets labeled with sentiment polarity.

## 🎯 Features

- **Text Preprocessing**: Removes special characters, converts to lowercase, and applies stemming
- **TF-IDF Vectorization**: Converts text data into numerical features
- **Logistic Regression Model**: Binary classification for sentiment analysis
- **Model Persistence**: Saves trained model and vectorizer for future use
- **High Accuracy**: Achieves good performance on the test dataset

## 📁 Project Structure

```
Sentiment_Analysis(Basic)/
├── README.md                           # This file
├── SentimentAnalysis.ipynb             # Main Jupyter notebook with analysis
├── trained_model.sav                   # Saved Logistic Regression model
├── vectorizer.pkl                      # Saved TF-IDF vectorizer
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Kaggle account and API credentials

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd LR_pickle
   ```

2. **Install required packages**
   ```bash
   pip install kaggle pandas numpy scikit-learn nltk
   ```

3. **Set up Kaggle credentials**
   - Place your `kaggle.json` file in the project directory
   - The notebook will automatically configure Kaggle API access

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## 📊 Dataset

The project uses the **Sentiment140 dataset** from Kaggle, which contains:
- 1.6 million tweets
- Binary sentiment labels (0 = negative, 4 = positive, converted to 0/1)
- Columns: target, id, date, flag, user, text

## 🔧 Usage

### Running the Analysis

1. Open `SentimentAnalysis.ipynb` in Jupyter Notebook
2. Run all cells sequentially to:
   - Download and extract the dataset
   - Preprocess the text data
   - Train the Logistic Regression model
   - Evaluate model performance
   - Save the trained model and vectorizer

### Using the Saved Model

The trained model and vectorizer are saved as:
- `trained_model.sav` - The trained Logistic Regression classifier
- `vectorizer.pkl` - The fitted TF-IDF vectorizer

You can load and use these files for making predictions on new text data.

## 🧠 Model Details

### Preprocessing Pipeline
1. **Text Cleaning**: Remove special characters and convert to lowercase
2. **Tokenization**: Split text into individual words
3. **Stop Word Removal**: Remove common English stop words
4. **Stemming**: Apply Porter Stemmer to reduce words to root form

### Model Architecture
- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF Vectorization
- **Training Split**: 80% training, 20% testing
- **Stratification**: Maintains class balance in splits

### Performance Metrics
- Training Accuracy: Evaluated on training data
- Test Accuracy: Evaluated on held-out test data

## 📈 Key Components

### Text Preprocessing Function
```python
def stemming(content):
    stm_cnt = re.sub('[^a-zA-Z]',' ',content)
    stm_cnt = stm_cnt.lower()
    stm_cnt = stm_cnt.split()
    stm_cnt = [stemed.stem(word) for word in stm_cnt if not word in stopwords.words('english')]
    stm_cnt = ' '.join(stm_cnt)
    return stm_cnt
```

### Model Training
```python
vectorizer = TfidfVectorizer()
model = LogisticRegression(max_iter=1000)
```


---

**Note**: Make sure to keep your Kaggle API credentials secure and never commit them to version control. 

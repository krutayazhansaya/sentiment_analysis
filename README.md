# 📊 Sentiment Analysis on YouTube Comments  

This project focuses on performing **Sentiment Analysis** on real YouTube comments.  
The goal was to build, evaluate, and compare different **Machine Learning models** and a **pre-trained Transformer model** (`distilbert-base-uncased-finetuned-sst-2-english`) to classify comments as **Positive** or **Negative**.  

---

## 📂 Dataset
- File: `comments_data.csv`  
- Contains user comments from YouTube with associated sentiment labels (`positive` / `negative`).  
- Size: ~XX rows  

### Example:
| Comment                         | Sentiment |
|---------------------------------|-----------|
| "This video is amazing!"        | positive  |
| "I really disliked the content" | negative  |

---

## 🔎 Exploratory Data Analysis (EDA)
- Removed missing values and duplicates  
- Tokenized and cleaned comments (stopwords, punctuation, lowercasing)  
- **Visualizations:**
  - Distribution of sentiment classes  
  - WordCloud for positive and negative comments  
  - Comment length distribution  

---

## ⚙️ Data Preparation
- Vectorization methods:
  - **Bag-of-Words (BoW)**
  - **TF-IDF**
- Train-test split: **80% training / 20% testing**

---

## 🤖 Machine Learning Models
We trained and evaluated the following models:  

- Logistic Regression  
- Naive Bayes  
- Random Forest  
- Support Vector Machine (SVM)  

### Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## 🔬 Model Results (Baseline ML)
		Accuracy	Precision	Recall	F1 Score
## Model Performance Comparison

| Vectorizer | Model                  | Accuracy | Precision | Recall  | F1 Score |
|------------|------------------------|----------|-----------|---------|----------|
| **BoW**    | Logistic Regression    | 0.893832 | 0.887795  | 0.893832 | 0.885540 |
|            | Random Forest          | 0.840374 | 0.831881  | 0.840374 | 0.793744 |
|            | Naive Bayes            | 0.883738 | 0.875916  | 0.883738 | 0.876554 |
|            | Support Vector Machine | 0.887850 | 0.882471  | 0.887850 | 0.884118 |
| **TF-IDF** | Logistic Regression    | 0.859065 | 0.858180  | 0.859065 | 0.826850 |
|            | Random Forest          | 0.838131 | 0.846273  | 0.838131 | 0.782965 |
|            | Naive Bayes            | 0.819813 | 0.762973  | 0.819813 | 0.741448 |
|            | Support Vector Machine | 0.888224 | 0.882825  | 0.888224 | 0.876112 |

---

## 🔁 Cross-Validation & Hyperparameter Tuning
For the **top 2 models** (Logistic Regression, SVM):  
- Applied **5-fold cross-validation**  
- Used **Grid Search** for hyperparameter tuning  

✅ Best model after tuning: **Bag-of-Words + Logistic Regression**  
- Best F1-score: **0.933**

---

## 🤗 Pre-trained Model (HuggingFace)
We also evaluated `distilbert-base-uncased-finetuned-sst-2-english`:  

| Model    | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| DistilBERT | 0.83   | 0.97      | 0.81   | 0.89     |

---

## 📊 Comparison
- **Logistic Regression (BoW)** achieved **higher F1-score (0.93)** compared to DistilBERT (0.89).  
- **BERT** has strong precision, but lower recall (misses some positive/negative cases).  
- **Logistic Regression** is lightweight, interpretable, and very effective for this dataset.  

---

## 🏆 Conclusion
- Best performing model: **Bag-of-Words + Logistic Regression**  
- Outperforms pre-trained BERT on this dataset.  
- Shows that with smaller, well-cleaned datasets, **classical ML methods can rival Transformers**.  

---


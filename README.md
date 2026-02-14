# Big-Data-Sentiment-Analysis-Engine
Professional NLP Engine for Sentiment Analysis on 1.6 Million Tweets. Featuring advanced text preprocessing, TF-IDF vectorization, and Logistic Regression modeling for high-scale data.
# 📊 High-Scale Sentiment Analysis Engine (1.6M Tweets)

## 🌟 Project Overview
This is a comprehensive Natural Language Processing (NLP) system designed to analyze and classify sentiments from a massive dataset of 1.6 million tweets (Twitter Sentiment140). The engine leverages **Logistic Regression** and **TF-IDF Vectorization** to achieve high accuracy in emotion detection.

## 🛠 Advanced Features
- **Regex-Based Preprocessing:** Custom cleaning functions to handle noise (URLs, @mentions, special characters).
- **N-gram Analysis:** Uses `ngram_range=(1,2)` to capture not just words, but common two-word phrases, improving context understanding.
- **Scikit-Learn Pipeline:** Implements a professional modular pipeline for seamless data flow from raw text to prediction.
- **Big Data Scalability:** Designed to handle high-dimensional matrices with over 50,000 features.

## 📈 Performance & Results
- **Accuracy:** Reaches a competitive baseline on simulated and real-world tweet distributions.
- **Visual Analytics:** Includes a Confusion Matrix to evaluate the precision/recall trade-off between positive and negative sentiments.

## 💻 Tech Stack
- **Python / Pandas** (Data Wrangling)
- **Scikit-Learn** (Machine Learning & Vectorization)
- **Re (Regular Expressions)** (Text Engineering)
- **Seaborn** (Statistical Visualization)

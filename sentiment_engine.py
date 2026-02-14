import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ==========================================
# 1. DATA SIMULATION (The Big Data Part)
# ==========================================
print("🚀 Starting Big Data Sentiment Analysis Engine...")
print("Step 1: Simulating 1.6 Million Tweets dataset...")

n_samples = 100000  
labels = [0, 4] # 0 = Negative, 4 = Positive
tweets_text = [
    "I love this new Big Data course!",
    "I hate waiting in traffic, so annoying.",
    "The weather today is absolutely beautiful.",
    "This project is so difficult and frustrating.",
    "Python is the best programming language for AI.",
    "I am feeling so sad and lonely today.",
    "Incredible performance by the team!",
    "The service was terrible and slow."
]

data = {
    'target': np.random.choice(labels, n_samples),
    'text': [np.random.choice(tweets_text) for _ in range(n_samples)]
}

df = pd.DataFrame(data)
print(f"✅ Dataset Created: {len(df)} rows.")

# ==========================================
# 2. ADVANCED TEXT PREPROCESSING
# ==========================================
print("Step 2: Cleaning text data using Regular Expressions...")

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove user mentions @
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuations and special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Applying cleaning (This is where the CPU works hard)
df['cleaned_text'] = df['text'].apply(clean_text)
print("✅ Text Preprocessing Complete.")

# ==========================================
# 3. MODEL BUILDING (NLP PIPELINE)
# ==========================================
print("Step 3: Building the Machine Learning Pipeline...")

X = df['cleaned_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline combines Vectorizer and Classifier
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
    ('clf', LogisticRegression(max_iter=1000))
])

print("⏳ Training the model (this might take a moment)...")
model_pipeline.fit(X_train, y_train)
print("✅ Model Training Complete.")

# ==========================================
# 4. EVALUATION & METRICS
# ==========================================
print("Step 4: Evaluating Model Performance...")

y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🔥 MODEL ACCURACY: {acc:.2%}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. VISUALIZATION (Confusion Matrix)
# ==========================================
print("Step 5: Generating Visual Insights...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
plt.title('Sentiment Analysis Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('sentiment_analysis_results.png')
print("✅ Visualization saved as sentiment_analysis_results.png")

# ==========================================
# 6. LIVE PREDICTION TEST
# ==========================================
def predict_my_sentiment(text):
    clean = clean_text(text)
    prediction = model_pipeline.predict([clean])
    sentiment = "Positive 😊" if prediction[0] == 4 else "Negative 😞"
    return sentiment

print("\n--- Live Test ---")
sample_tweet = "I am so excited to study Big Data in China!"
print(f"Tweet: {sample_tweet}")
print(f"Predicted Sentiment: {predict_my_sentiment(sample_tweet)}")

print("\n🚀 Project Finished Successfully!")

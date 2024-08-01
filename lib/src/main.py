import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppresses most messages, showing only errors
tf.get_logger().setLevel('ERROR')         # Ensures that only errors are outputted
# reading cleaned twitter dataset
df = pd.read_csv('../data/Cleaned_Tweets_Dataset.csv')

# tokenizing tweets with TF-IDF
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
X = tfidf.fit_transform(df['text'].values.astype('U'))
y = df['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# loading logistic regression trained model (Eli's path: '../models/logistic_regression_model.pkl')
log_reg = joblib.load('../models/logistic_regression_model.pkl') 

# loading support vector classification trained model (Eli's path: '../models/support_vector_classification_model.pkl')
svc = joblib.load('../models/support_vector_classification_model.pkl')

# Assuming RandomForest is added
rf_model = joblib.load('../models/random_forest_model.pkl')  # Load RandomForest model

# prediction
log_prediction = log_reg.predict(X_test)
svc_prediction = svc.predict(X_test)
rf_prediction = rf_model.predict(X_test)  # RandomForest prediction

# Prepare data for LSTM (requires sequence data)
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(df['text'].astype(str))
sequences = tokenizer.texts_to_sequences(df['text'].astype(str))
X_seq = pad_sequences(sequences, maxlen=100)  # Using max length of 100, adjust as needed
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y, test_size=0.3, random_state=42)

# Load LSTM model
lstm_model = load_model('../models/sentiment_lstm_model.keras')

# LSTM prediction
lstm_prediction = lstm_model.predict(X_test_seq)
lstm_prediction = (lstm_prediction > 0.5).astype(int)

# Calculate accuracy scores
log_accuracy = accuracy_score(log_prediction, y_test)
svc_accuracy = accuracy_score(svc_prediction, y_test)
rf_accuracy = accuracy_score(rf_prediction, y_test)  # RandomForest accuracy
lstm_accuracy = accuracy_score(lstm_prediction, y_test_seq)

print()

# Print accuracy scores
print(f"Logistic Regression Accuracy Score: {log_accuracy}")
print(f"Support Vector Classification Accuracy Score: {svc_accuracy}")
print(f"Random Forest Accuracy Score: {rf_accuracy}")
print(f"LSTM Model Accuracy Score: {lstm_accuracy}")

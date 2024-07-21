import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# reading cleaned twitter dataset
df = pd.read_csv('../data/Cleaned_Tweets_Dataset.csv')
# tokenizing tweets
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

# creating training and test values
X = tfidf.fit_transform(df['text'].values.astype('U'))
y = df['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# loading logistic regression trained model
log_reg = joblib.load('../models/logistic_regression_model.pkl')

# loading support vector classification trained model
svc = joblib.load('../models/support_vector_classification_model.pkl')

# prediction
log_prediction = log_reg.predict(X_test)
svc_prediction = svc.predict(X_test)

# accuracy scores
print(f"Logistic Regression Accuracy Score: {accuracy_score(log_prediction, y_test)}")
print(f"Support Vector Classification Accuracy Score: {accuracy_score(svc_prediction, y_test)}")
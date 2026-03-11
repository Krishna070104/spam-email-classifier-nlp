import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from preprocess import preprocess_text

# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding='latin-1')

# Convert labels to 0 and 1 (Category column has 'ham'/'spam')
# Note: Same mapping as train.py - ham=1, spam=0
df['label'] = df['Category'].map({'ham':1,'spam':0})

# Preprocess text (Message column contains the email text)
df['clean_message'] = df['Message'].apply(preprocess_text)

# Load model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

X = vectorizer.transform(df['clean_message']).toarray()
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print("\nClassification Report\n")
print(classification_report(y_test,pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,pred))


import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess_text

#load dataset
data = pd.read_csv("dataset/spam.csv",encoding='latin-1')
# convert labels to 0 and 1 (Category column has 'ham'/'spam')
data['label'] = data['Category'].map({'ham':1,'spam':0})
#preprocess text (Message column contains the email text)
data['cleaned_text'] = data['Message'].apply(preprocess_text)
# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['cleaned_text']).toarray()
y = data['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Model
model = MultinomialNB()

model.fit(X_train,y_train)
# Save model
joblib.dump(model,"model/spam_model.pkl")
joblib.dump(vectorizer,"model/vectorizer.pkl")

print("Model trained and saved successfully.")
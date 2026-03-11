import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

ps = PorterStemmer()
st_words = set(stopwords.words('english'))
def preprocess_text(txt):
    # Handle NaN values and non-string types
    if pd.isna(txt) or not isinstance(txt, str):
        return ""
    txt = txt.lower()# Convert to lowercase
    txt = re.sub('[^a-zA-Z]',' ' ,txt) # Remove special characters and numbers
    words = txt.split() #tokenization
    words = [ps.stem(word) for word in words if word not in  st_words]# Remove stop words and stemming
    return " ".join(words)

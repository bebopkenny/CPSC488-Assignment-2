import numpy as np
import pandas as pd
from collections import Counter
import re
import nltk
# Download NLTK resources if not already present
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load a text dataset (documents) into a dataframe
docs_df = pd.read_csv('reviews.csv') # docs is a dataframe with columns 'text' and 'label'

# Various text preprocessing packages available:
# NLTK, Gensim, SpaCy, TextBlob, HuggingFace Transformers, etc.

####### Method 1: Document-Term Matrix (DTM) from scratch ########
# For tokenization using NLTK
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens

# tokenize the documents
docs_df['tokens'] = docs_df['text'].apply(preprocess)

# Drop rows with empty tokens
tokenized_docs = [tokens for tokens in docs_df['tokens'] if len(tokens) > 0]
print("number of documents:", len(docs_df))
print("number of non-empty documents:", len(tokenized_docs))
print("documents with tokens")
print(tokenized_docs)

# count words for each document
doc_counts = [Counter(doc) for doc in tokenized_docs]

# Build a DTM DataFrame
dtm_df = pd.DataFrame([{word: counts.get(word, 0) for word in set().union(*doc_counts)} for counts in doc_counts])
print(dtm_df)
#exit()

####### Method 2: Document-Term Matrix (DTM) using CountVectorizer ########
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer

# Initialize vectorizer
# CountVectorizer does both tokenization and counting
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Fit (identifies unique terms) and transform (vectorizes the terms)
# it can take an entire text dataset instead of one document
X = vectorizer.fit_transform(docs_df['text'])
# it returns (row id, column id) frequency with non-zero entries where column id is the term index
print(X)
# Convert a vectorized object to DataFrame (table format)
dtm_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(dtm_df)
#exit()

####### Method 3: Document-Term Matrix (DTM) using Gensim ########
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary

# tokenization
tokenized_docs = [
    [word for word in simple_preprocess(doc) if word not in STOPWORDS]
    for doc in docs_df['text']]

# Create dictionary
dictionary = Dictionary(tokenized_docs)
# Create bag-of-words in Gensim format where each doc2bow is a list of (word_id, count) pairs
bow = [dictionary.doc2bow(text) for text in tokenized_docs]
# Convert to Document-Term Matrix (pandas DataFrame)
dtm_df = pd.DataFrame([{dictionary[id]: count for id, count in doc} for doc in bow]).fillna(0).astype(int)
print(dtm_df)
#exit()

####### Method 4: Document-Term Matrix (DTM) using TfidfVectorizer ########
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
# Fit and transform
X = vectorizer.fit_transform(docs_df['text'])
# Convert to DataFrame
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
# Print TF-IDF DTM rounded to 3 decimal places
print(tfidf_df.round(3))

####### Method 5: Custom/curated Document-Term Matrix (DTM) ########
# You can create a custom matrix with additional features for the curated terms like these
curated_terms = ['good', 'bad', 'excellent', 'poor', 'great', 'terrible']

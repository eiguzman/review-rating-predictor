import gzip
import json
import string
import random
import emoji
import re
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from collections import defaultdict
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import os
import joblib

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

trainRatings = []
testRatings = []
num_obs = 100_000
curr = 0

#set seed
random.seed(42)

for i in parse('review-California_10.json.gz'):
    if curr >= num_obs:
        break
    luck = random.randint(1,10)
    if luck < 6:
        trainRatings.append(i)
    if luck == 6:
        testRatings.append(i)
    curr += 1

punct = string.punctuation
punct = ''.join([punct] + ['\u2019', '\u201c', '\u201d'])

def add_spaces_after_emojis(text):
    pattern = r'(:\w+:)'
    result = re.sub(pattern, r' \1 ', text)
    return result

def add_spaces_after_newline(text):
    pattern = r'(\n)'
    result = re.sub(pattern, r' ', text)
    return result

def textractor(ratings):
    for d in enumerate(ratings):
        if d[1]['text'] is None:
            r = ''
        else:
            r = d[1]['text'].lower()
            parts = r.split('{original}')
            r = parts[0].strip()
            parts = r.split('(original)')
            r = parts[0].strip()
            r = ''.join([c for c in r if c not in punct])
            r = ' '.join(add_spaces_after_newline(r).split())
        ratings[d[0]]['text'] = r
    return ratings

trainRatings = textractor(trainRatings)
testRatings = textractor(testRatings)

unis = {"\u2605": ":u_black_star:", "\u2606": ":u_white_star:", "\u2661": ":u_white_heart:", "\u2665": ":u_black_heart:"}

def make_csv(path, data):
    predictions = open(path, 'w', encoding= 'utf-8')
    predictions.write('User_ID,Business_ID,Time,Text,Rating\n')
    for i in data:
        u = i['user_id']
        b = i['gmap_id']
        tm = str(i['time'])
        tx = i['text']
        r = str(i['rating'])
        for j in unis:
            tx = tx.replace(j, unis[j])
        if any(char in emoji.EMOJI_DATA for char in tx):
            tx_demojize = emoji.demojize(tx)
            res = add_spaces_after_emojis(tx_demojize).split()
            tx = ' '.join(res)
        results = f'{u},{b},{tm},"{tx}",{r}\n'
        predictions.write(results)
    predictions.close()
    return

make_csv('review-California-train.csv', trainRatings)
make_csv('review-California-test.csv', testRatings)

df = pd.read_csv('review-California-train.csv', encoding= 'utf-8')

wordCount = defaultdict(int)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Bag-of-words processor with N-gram logic
def process_text(text):
    if pd.isna(text):
        return np.nan
    words = ' '.join(np.array([stemmer.stem(w) for w in text.split() if w not in stop_words]))
    n1 = words.split()
    n2 = [' '.join(x) for x in list(zip(n1[:-1],n1[1:]))]
    # Due to large data sizes, limit N-grams to 2
    # n3 = [' '.join(x) for x in list(zip(n1[:-2],n1[1:-1],n1[2:]))]
    # n4 = [' '.join(x) for x in list(zip(n1[:-3],n1[1:-2],n1[2:-1],n1[3:]))]
    # n5 = [' '.join(x) for x in list(zip(n1[:-4],n1[1:-3],n1[2:-2],n1[3:-1],n1[4:]))]
    for w in n1 + n2:
        wordCount[w] += 1
    return n1 + n2

df2 = df.copy()
df2['Bag_of_Words'] = df['Text'].apply(process_text)

df_no_text = df2[df2["Text"].isna()]
df_with_text = df2[~df2["Text"].isna()]

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
words = np.array([x[1] for x in counts[:3000]])

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
    feat = [0]*len(words)
    for w in datum:
        if w in words:
            feat[wordId[w]] += 1
    feat.append(1) # offset
    return np.array(feat)

X = np.array([feature(d) for d in df_with_text["Bag_of_Words"]])
y = np.array([d for d in df_with_text["Rating"]])

#50k training data, top 3000 words

model_filename = 'best_ridge_model.pkl'

# Check if the model file exists
if os.path.exists(model_filename):
    # Load the existing model
    best_ridge = joblib.load(model_filename)
    print("Loaded existing model from", model_filename)
else:
    # Define parameter grid for regularization strength
    # param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    # The best alpha is 100 out of 5 CV attempts
    param_grid = {'alpha': [100.0]}
    
    # Set up 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize Ridge regression model
    ridge = Ridge(fit_intercept=False)
    
    # Use GridSearchCV for hyperparameter tuning with cross-validation
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit models to find the best alpha
    grid_search.fit(X, y)
    
    # Retrieve the best model
    best_ridge = grid_search.best_estimator_
    
    # Optional: print best parameters
    print("Best alpha:", grid_search.best_params_['alpha'])
    
    # Final training on entire dataset with best alpha
    best_ridge.fit(X, y)
    
    # Save the best model to a file
    joblib.dump(best_ridge, model_filename)

# Make predictions
predictions = best_ridge.predict(X)

# Round predictions to nearest integer within 1-5
ypred = [min(5, max(1, round(p))) for p in predictions]

# Function to compute Mean Squared Error
def MSE(preds, labels):
    return mean_squared_error(labels, preds)

# Compute and print final MSE
final_mse = MSE(ypred, y)
print("Final MSE on training data:", final_mse)

df_test = pd.read_csv('review-California-test.csv', encoding= 'utf-8')

wordCount = defaultdict(int)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

df2_test = df_test.copy()
df2_test['Bag_of_Words'] = df_test['Text'].apply(process_text)

test_no_text = df2_test[df2_test["Text"].isna()]
test_with_text = df2_test[~df2_test["Text"].isna()]

X_test = np.array([feature(d) for d in test_with_text["Bag_of_Words"]])
y_yest = np.array([d for d in test_with_text["Rating"]])

predictions = best_ridge.predict(X_test)

ypred_test = [5 if p > 5 else p for p in predictions]
ypred_test = [1 if p < 1 else p for p in ypred_test]

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(ypred_test,labels)]
    return sum(differences) / len(differences)

print("MSE on test data:", MSE(ypred_test, y_yest))
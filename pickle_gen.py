#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:45:44 2024

@author: aashay
"""

import pandas as pd
df = pd.read_csv('sentiment_analyzed.csv', index_col = 0)
df.dropna(inplace = True)
df.reset_index(inplace = True)

cuisine_list = df['cuisines'].unique().tolist()
uniq_cuisines = set()
for cuisine in cuisine_list:
    cuisine_temp = [x.strip() for x in cuisine.split(',')]
    uniq_cuisines.update(cuisine_temp)

cuisine_uniq_list = sorted(uniq_cuisines)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2),stop_words='english', min_df=0.0)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['reviews_list'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

import pickle
pickle.dump(cosine_similarities, open('cos_sim.pkl', 'wb'))
pickle.dump(cuisine_uniq_list, open('cuisine_uniq.pkl', 'wb'))


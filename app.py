# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import pickle
import streamlit as st 

def load_():
    global df
    global cosine_similarities
    global cuisine_uniq
    
    if 'df' not in globals():
        df = pd.read_csv('sentiment_analyzed.csv')
    
    if 'cosine_similarities' not in globals(): 
        with open('cos_sim.pkl', 'rb') as file:
            cosine_similarities = pickle.load(file)
            
    if 'cuisine_uniq' not in globals():
        with open('cuisine_uniq.pkl', 'rb') as file:
            cuisine_uniq = pickle.load(file)
    
    return

df = pd.read_csv('sentiment_analyzed.csv')

with open('cos_sim.pkl', 'rb') as file:
    cosine_similarities = pickle.load(file)
    
with open('cuisine_uniq.pkl', 'rb') as file:
    cuisine_uniq = pickle.load(file)
    
def recommend(name, cuisine = None, cosine_similarities = None):
    if cosine_similarities is None:
        cosine_similarities = globals()['cosine_similarities']
        
    df_temp = df.copy(deep = True)
    if cuisine:
        df_temp = df[df['cuisines'].str.contains(cuisine)]
    
    idx = df[df['name'] == name].index[0]
    similar_idx = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)

    top100idx = list(similar_idx.iloc[1:100].index)

    df_new = df_temp[df_temp.index.isin(top100idx)][['name', 'location', 'rate', 'votes', 'cost', 'cuisines', 'super_score']]
    df_new.drop_duplicates(subset = 'name', inplace = True)
    df_new = df_new.sort_values(by = 'super_score', ascending = False).head(5)
    print('Top Restaurants similar to %s: ' % (name))

    return df_new


def show_page():
    global df
    global cuisine_uniq
    rest_name_uniq = ['<select>'] + df['name'].unique().tolist()
    st.title("Restaurant Recommendation System")
    st.write("""### This recommender system will suggest other restaurants based on your current favourite""")
    
    rest_name = st.selectbox('Restaurant Name', rest_name_uniq)

    cuisine_entry = st.selectbox('cuisine', options = [None] + cuisine_uniq)
    
    if st.button('Recommend'):
        if rest_name != '<select>':
            res = recommend(name = rest_name, cuisine = cuisine_entry)
            st.write("Recommendations: ")
            st.dataframe(res)
        
    
    


load_()
show_page()
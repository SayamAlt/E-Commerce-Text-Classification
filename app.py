#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import joblib
from helper_prabowo_ml import clean_html, remove_special_characters, punct, lower, email_address, remove_digits, remove_links, remove_, removeStopWords, non_ascii


# In[2]:


model = joblib.load('backup_model.pkl')


# In[3]:


vectorizer = joblib.load('tfidf_vectorizer.pkl')


# In[6]:


label_mapping = {0: 'Books', 1: 'Clothing & Accessories', 2: 'Electronics', 3: 'Household'}


# In[7]:


def main():
    st.title('E-Commerce Category Prediction')
    prod_description = st.text_area(label="Give details of your product:")
    prod_description = lower(prod_description)
    prod_description = punct(prod_description)
    prod_description = email_address(prod_description)
    prod_description = non_ascii(prod_description)
    prod_description = remove_(prod_description)
    prod_description = remove_digits(prod_description)
    prod_description = remove_links(prod_description)
    prod_description = remove_special_characters(prod_description)
    prod_description = removeStopWords(prod_description)
    prod_description = clean_html(prod_description)
    prod_description = vectorizer.transform([prod_description])
    
    if st.button("Predict"):
        pred = model.predict(prod_description)[0]
        output = str(label_mapping[pred])
        st.success(f"The category of your e-commerce product is {output.lower()}.")


# In[8]:


if __name__ == "__main__":
    main()


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
from nltk.stem import PorterStemmer, WordNetLemmatizer
from symspellpy import Verbosity, SymSpell
from helper_prabowo_ml import remove_, remove_digits, remove_special_characters, remove_links, removeStopWords, punct, clean_html, email_address, non_ascii, lower


# In[2]:


app = Flask(__name__)


# In[3]:


model = joblib.load('backup_model.pkl')
model


# In[4]:


tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_vectorizer


# In[5]:


@app.route("/")
def home():
    return render_template('index.html')


# In[6]:


stemmer = PorterStemmer()

def stem_words(text):
     return ' '.join(stemmer.stem(word) for word in text.split())


# In[7]:


lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())


# In[8]:


spelling_corrector = SymSpell()

def correct_spellings(text):
    corrected_tokens = []
    
    for token in text.split():
        x = spelling_corrector.lookup(token,Verbosity.CLOSEST,max_edit_distance=2,include_unknown=True)[0].__str__()
        y = x.split(',')[0]
        corrected_tokens.append(y)
    
    return ' '.join(corrected_tokens)


# In[9]:


label_mapping = {0: 'Books', 1: 'Clothing & Accessories', 2: 'Electronics', 3: 'Household'}
label_mapping


# In[10]:


@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        prod_description = request.form['product_description']
        prod_description = lower(prod_description)
        prod_description = remove_(prod_description)
        prod_description = remove_digits(prod_description)
        prod_description = remove_links(prod_description)
        prod_description = remove_special_characters(prod_description)
        prod_description = removeStopWords(prod_description)
        prod_description = punct(prod_description)
        prod_description = non_ascii(prod_description)
        prod_description = email_address(prod_description)
        prod_description = clean_html(prod_description)
        prod_description = stem_words(prod_description)
        prod_description = lemmatize_words(prod_description)
        prod_description = correct_spellings(prod_description)
        vectorized_prod_description = tfidf_vectorizer.transform([prod_description])
        pred = model.predict(vectorized_prod_description)[0]
        output = str(label_mapping[pred]).lower()
        return render_template('index.html',prediction_text=f'The e-commerce category based on specified product description is {output}.')


# In[ ]:


if __name__ == '__main__':
    app.run(port=8080)


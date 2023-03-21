#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import pickle 
import string
import nltk
import numpy as np
from flask import Flask, render_template, request
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import requests
#nltk.download('punkt')
#nltk.download('wordnet')

app = Flask(__name__)
model = pickle.load(open('best_SVC_Model.pkl','rb')) #read mode
vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb')) #read mode

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        
        #access the data from form
        text_input = request.form["text_input"] # the html code for the text input should also be text_input
        
        preprocessed_input = preprocess(text_input)
        
        # Initialize the TF-IDF vectorizer
        #vectorizer = TfidfVectorizer()

        # Vectorize the preprocessed input using TF-IDF
        vectorized_input = vectorizer.transform([preprocessed_input])

        # Use the pre-trained model to make predictions
        prediction = model.predict(vectorized_input)

        # Map the prediction to a sentiment label
        sentiment = 'positive' if prediction[0] == 1 else 'negative'
        
        return render_template("index.html", prediction_text='The sentiment of your review is $ {}'.format(sentiment))
        
def preprocess(text):
    # Remove punctuation and lowercase the text
    if isinstance(text, str) and len(text) > 0:
        text = text.translate(str.maketrans('', '', string.punctuation))#.lower()
    else:
        print("Invalid input")

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a string
    text = ' '.join(words)

    return text
        
       
if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8080)


# In[ ]:





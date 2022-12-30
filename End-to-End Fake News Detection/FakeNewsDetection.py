#Import Important Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#Data Collection
data = pd.read_csv("data/fake_or_real_news.csv")

#Feature Selection
x = np.array(data["title"]) #feature
y = np.array(data["label"]) # target

cv = CountVectorizer()
x = cv.fit_transform(x)

#Spliting Data
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42)

#Training a machine learning model
model = MultinomialNB()
model.fit(xtrain, ytrain)


import streamlit as st
st.title("Fake News Detection System")

def fakenewsdetection():
    user = st.text_area("Enter Any News headline")
    if len(user) < 1:
        st.write(" ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = model.predict(data)
        st.title(a)
fakenewsdetection()

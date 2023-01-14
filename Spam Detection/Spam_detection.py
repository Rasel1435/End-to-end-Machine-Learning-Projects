# import necessary library
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Spliting Data
from sklearn.model_selection import train_test_split

# Choosing Model & Training The Model
from sklearn.naive_bayes import MultinomialNB

# Data Collection
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding='latin-1')

# Feature Selection
data = data[["class", "message"]]
x = np.array(data["message"])
y = np.array(data["class"])

# Choosing Model & Training The Model
cv = CountVectorizer()
x = cv.fit_transform(x) # fit the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# another model 
clf = MultinomialNB()
clf.fit(x_train, y_train)

# import streamlit 
import streamlit as st
st.title("Spam Detection System")
def spamdetection():
    user = st.text_area("Enter any Message or Email: ")
    if len(user) < 1:
        st.write(" ")
    else:
        sample = user
        data = cv.transform[sample].toarray()
        a = clf.predict(data)
        st.title(a)
spamdetection()
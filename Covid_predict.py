# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#Prediction Function
def prediction(model, Sex, RaceEthnicity,Age,Dia):
  comp = model.predict([[Sex, RaceEthnicity,Age,Dia]])
  comp = comp[0]
  if comp == 0:
    return "low"
  elif comp == 1:
    return "high"
# Configuring the home page.
st.set_page_config(page_title = "COVID Complication Prediction", 
                          page_icon = ":virus:", 
                          layout = 'centered', 
                          initial_sidebar_state = 'auto')
# Add title on the main page and in the sidebar.
st.title("COVID Complication Predictor")
#User Values    
st.sidebar.subheader("Select your values:")
st.sidebar.write('0 is Male and 1 is Female')
gender = int(st.sidebar.selectbox("Choose your gender: ",('0','1')))
st.sidebar.write("  Hispanic-1, Native American-2, Asian-3")
st.sidebar.write("African American-4, White-5, Other-6")
race = int(st.sidebar.selectbox("Choose your race: ",('1','2','3','4','5','6')))
st.sidebar.write("How old are you? Choose the closest number to your age")
age = int(st.sidebar.selectbox("Input Age",('0','5','15','25','35','45','55','65','75',85)))
st.sidebar.write('Do you have diabetes? Choose 0 if you dont and choose 1 if you do.')
diabetes = int(st.sidebar.selectbox("Input Diabetes history",('0','1')))  
filename_lr = 'lr_model.sav'

#Classifier
classifier = 'Logistic Regression'
if classifier == 'Logistic Regression':
    # If the user clicks the 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
    if st.sidebar.button('Classify'):
    	loaded_model_lr = pickle.load(open(filename_lr, 'rb'))
    	comp_lr = prediction(loaded_model_lr,gender,race,age,diabetes)
    	st.write("The Chance for COVID complication:", comp_lr)

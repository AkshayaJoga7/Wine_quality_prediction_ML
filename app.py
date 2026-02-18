import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("wine_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Wine Quality Prediction App")

inputs = []

for feature in [
    "fixed acidity","volatile acidity","citric acid",
    "residual sugar","chlorides","free sulfur dioxide",
    "total sulfur dioxide","density","pH",
    "sulphates","alcohol"
]:
    value = st.number_input(feature)
    inputs.append(value)

if st.button("Predict"):
    scaled = scaler.transform([inputs])
    result = model.predict(scaled)
    
    if result[0] == 1:
        st.success("Good Quality Wine 🍷")
    else:
        st.error("Bad Quality Wine")

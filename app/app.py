import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Dashboard Title
st.title("Malaysian Road Accident Analysis Dashboard")

# Project Header
st.header("WIE2003 Introduction to Data Science Project")

# Load Dataset
df = pd.read_excel("data/Cleaned_Road_Safety_Data_Fixed.xlsx")

# Dataset Preview
st.subheader("Dataset Preview")

st.dataframe(df.head())

# Road Death Trend Chart
st.subheader("Road Death Trend Over Time")

fig, ax = plt.subplots(figsize=(10,5)) # Create a figure and axis object for plotting

ax.plot(df['Year'], df['Road Deaths'], marker='o') # Plot the line chart with markers for each data point

ax.set_xlabel("Year")
ax.set_ylabel("Road Deaths")
ax.set_title("Road Death Trend")

st.pyplot(fig) # Display the plot in Streamlit 

# Project Description
st.write("""
This dashboard presents:
- Exploratory Data Analysis (EDA)
- Regression Modelling
- Road Death Predictions
- Data Visualizations
""")

st.success("Streamlit app is working successfully!")
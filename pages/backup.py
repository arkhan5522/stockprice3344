import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
st.set_page_config(page_title="Stock Price",layout="wide",page_icon="icon.jpg",)
st.title("Stock Price Prediction using Machine Learning")
st.write("Welcome to our Stock Price Prediction website, where we utilize advanced machine learning techniques to provide accurate and insightful forecasts for stock prices. In an ever-changing and volatile market, staying ahead of the curve is crucial for investors, traders, and financial professionals. Our platform aims to empower users with reliable predictions, enabling informed decision-making and potentially maximizing return")
st.image("icon.jpg")
with open('wave.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
st.sidebar.header('Choose Date from below')
start_date=st.sidebar.date_input('Start date',date (2023,1,1))
end_date=st.sidebar.date_input('End date',date (2024,1,1))
ticker_list= ["ABL","ABOT"]
ticker=st.selectbox('Select the Company',ticker_list)
if  ticker == "ABL":
   data=pd.read_excel("DATA SET/ABL_merged_final.xlsx")
if  ticker == "ABOT":
   data=pd.read_excel("DATA SET/ABOT_merged_final.xlsx")
st.write(data)

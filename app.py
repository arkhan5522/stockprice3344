import streamlit as st
import streamlit.components.v1 as com
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import pages
from datetime import date
import datetime
import requests
st.set_page_config(page_title="Stock Price",layout="wide",page_icon="icon.jpg")
with st.sidebar:
   selected=option_menu(
   menu_title="Pages",
   options=["home","graphs"],
   )
   if selected == "graphs":
    st.switch_page("pages/backup.py")
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://img.freepik.com/free-photo/white-geometrical-shapes-background_23-2148811541.jpg?w=1380");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
com.html("""
<H1> Stock Price Prediction using Machine Learning</H1>
         <div class="khansb">
Welcome to our Stock Price Prediction website, where we utilize advanced machine learning techniques to provide accurate and insightful forecasts for stock prices. In an ever-changing and volatile market, staying ahead of the curve is crucial for investors, traders, and financial professionals. Our platform aims to empower users with reliable predictions, enabling informed decision-making and potentially maximizing return  
      
            </div>
<style>
         
      html
         {
         margin: auto;
         overflow: auto;
        background: transparent;
         background-attachment: fixed;
         background-size: 400% 400%;
         color:black;
         font-size: large;
               }
H1{
         text-align: center;
         background-color: rgb(119 218 149);

}
         </style>
""")
st.image("icon.jpg")
ticker_list= ["ABL","ABOT","AGP","AICL","AIRLINK","AKBL","APL","ARPL","ATLH","ATRL","AVN","BAFL","BAHL","BNWM","BOP","CEPB","CNERGY","COLG","DAWH","DCR","DGKC","EFERT","EFUG","ENGRO","EPCL","FABL","FATIMA","FCCL","FCEPL","FFBL","FFC","FHAM","GADT","GATM","GHGL","GLAXO","HBL","HCAR","HGFA","HINOON","HMB","HUBC","IBFL","ILP","INDU","INIL","ISL","JDWS","JVDC","KAPCO","KEL","KOHC","KTML","LCI","LOTCHEM","LUCK","MARI","MCB","MEBL","MTL","MUGHAL","MUREB","NATF","NBP","NESTLE","NML","NRL","OGDC","PABC","PAEL","PAKT","PGLC","PIBTL","PIOC","POL","POML","PPL","PSEL","PSMC","PSO","PSX","PTC","RMPL","SCBPL","SEARL","SHEL","SHFA","SNGP","SRVI","SYS","TGL","THALL","TRG","UBL","UNITY","UPFL","YOUW"]
ticker=st.selectbox('Select the Company',ticker_list)
with st.sidebar:
   selected=option_menu(
   menu_title="selected company",
   options=[ticker])
   st.markdown(
    """
<style>
.sidebar .sidebar-content {
   background-color: #77da95;
    color: white;
}
.menu .container-xxl[data-v-5af006b8] {
    background-image: linear-gradient(#868F96 , #596164);
    border-radius: .5rem;
}
</style>
""",
    unsafe_allow_html=True,
)
with open('style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

if  ticker == "ABL":
   data=pd.read_excel("DATA SET/ABL_merged_final.xlsx")
if  ticker == "ABOT":
   data=pd.read_excel("DATA SET/ABOT_merged_final.xlsx") 
if  ticker == "AGP":
   data=pd.read_excel("DATA SET/AGP_merged_final.xlsx")
if  ticker == "AICL":
   data=pd.read_excel("DATA SET/AICL_merged_final.xlsx")
if  ticker == "AIRLINK":
   data=pd.read_excel("DATA SET/AIRLINK_merged_final( MSD).xlsx")
if  ticker == "AKBL":
   data=pd.read_excel("DATA SET/AKBL_merged_final.xlsx")
if  ticker == "APL":
   data=pd.read_excel("DATA SET/APL_merged_final.xlsx")
if  ticker == "ARPL":
   data=pd.read_excel("DATA SET/ARPL_merged_final.xlsx")
if  ticker == "ATLH":
   data=pd.read_excel("DATA SET/ATLH_merged_final.xlsx")
if  ticker == "ATRL":
   data=pd.read_excel("DATA SET/ATRL_merged_final.xlsx")
if  ticker == "AVN":
   data=pd.read_excel("DATA SET/AVN_merged_final.xlsx")
if  ticker == "BAFL":
   data=pd.read_excel("DATA SET/BAFL_merged_final.xlsx")
if  ticker == "BAHL":
   data=pd.read_excel("DATA SET/BAHL_merged_final.xlsx")
if  ticker == "BNWM":
   data=pd.read_excel("DATA SET/BNWM_merged_final.xlsx")
if  ticker == "BOP":
   data=pd.read_excel("DATA SET/BOP_merged_final.xlsx")
if  ticker == "CEPB":
   data=pd.read_excel("DATA SET/CEPB_merged_final.xlsx")
if  ticker == "CNERGY":
   data=pd.read_excel("DATA SET/CNERGY_merged_final.xlsx")
if  ticker == "COLG":
   data=pd.read_excel("DATA SET/COLG_merged_final.xlsx")
if  ticker == "DAWH":
   data=pd.read_excel("DATA SET/DAWH_merged_final.xlsx")
if  ticker == "DCR":
   data=pd.read_excel("DATA SET/DCR_merged_final.xlsx")
if  ticker == "DGKC":
   data=pd.read_excel("DATA SET/DGKC_merged_final.xlsx")
if  ticker == "EFERT":
   data=pd.read_excel("DATA SET/EFERT_merged_final.xlsx")
if  ticker == "EFUG":
   data=pd.read_excel("DATA SET/EFUG_merged_final.xlsx")
if  ticker == "ENGRO":
   data=pd.read_excel("DATA SET/ENGRO_merged_final.xlsx")
if  ticker == "EPCL":
   data=pd.read_excel("DATA SET/EPCL_merged_final.xlsx")
if  ticker == "FABL":
   data=pd.read_excel("DATA SET/FABL_merged_final.xlsx")
if  ticker == "FATIMA":
   data=pd.read_excel("DATA SET/FATIMA_merged_final.xlsx")
if  ticker == "FCCL":
   data=pd.read_excel("DATA SET/FCCL_merged_final.xlsx")
if  ticker == "FCEPL":
   data=pd.read_excel("DATA SET/FCEPL_merged_final.xlsx")
if  ticker == "FFBL":
   data=pd.read_excel("DATA SET/FFBL_merged_final(MSD).xlsx")
if  ticker == "FFC":
   data=pd.read_excel("DATA SET/FFC_merged_final.xlsx")
if  ticker == "FHAM":
   data=pd.read_excel("DATA SET/FHAM_merged_final.xlsx")
if  ticker == "GADT":
   data=pd.read_excel("DATA SET/GADT_merged_final.xlsx")
if  ticker == "GATM":
   data=pd.read_excel("DATA SET/GATM_merged_final.xlsx")
if  ticker == "GHGL":
   data=pd.read_excel("DATA SET/GHGL_merged_final.xlsx")
if  ticker == "GLAXO":
   data=pd.read_excel("DATA SET/GLAXO_merged_final.xlsx")
if  ticker == "HBL":
   data=pd.read_excel("DATA SET/HBL_merged_final.xlsx")
if  ticker == "HCAR":
   data=pd.read_excel("DATA SET/HCAR_merged_final.xlsx")
if  ticker == "HGFA":
   data=pd.read_excel("DATA SET/HGFA_merged_final(MSD).xlsx")
if  ticker == "HINOON":
   data=pd.read_excel("DATA SET/HINOON_merged_final.xlsx")
if  ticker == "HMB":
   data=pd.read_excel("DATA SET/HMB_merged_final.xlsx")
if  ticker == "HUBC":
   data=pd.read_excel("DATA SET/HUBC_merged_final(MSD).xlsx")
if  ticker == "IBFL":
   data=pd.read_excel("DATA SET/IBFL_merged_final(LD).xlsx")
if  ticker == "ILP":
   data=pd.read_excel("DATA SET/ILP_merged_final.xlsx")
if  ticker == "INDU":
   data=pd.read_excel("DATA SET/INDU_merged_final.xlsx")
if  ticker == "INIL":
   data=pd.read_excel("DATA SET/INIL_merged_final.xlsx")
if  ticker == "ISL":
   data=pd.read_excel("DATA SET/ISL_merged_final.xlsx")
if  ticker == "JDWS":
   data=pd.read_excel("DATA SET/JDWS_merged_final.xlsx")
if  ticker == "JVDC":
   data=pd.read_excel("DATA SET/JVDC_merged_final.xlsx")
if  ticker == "KAPCO":
   data=pd.read_excel("DATA SET/KAPCO_merged_final.xlsx")
if  ticker == "KEL":
   data=pd.read_excel("DATA SET/KEL_merged_final.xlsx")
if  ticker == "KOHC":
   data=pd.read_excel("DATA SET/KOHC_merged_final.xlsx")
if  ticker == "KTML":
   data=pd.read_excel("DATA SET/KTML_merged_final.xlsx")
if  ticker == "LCI":
   data=pd.read_excel("DATA SET/LCI_merged_final.xlsx")
if  ticker == "LOTCHEM":
   data=pd.read_excel("DATA SET/LOTCHEM_merged_final.xlsx")
if  ticker == "LUCK":
   data=pd.read_excel("DATA SET/LUCK_merged_final.xlsx")
if  ticker == "MARI":
   data=pd.read_excel("DATA SET/MARI_merged_final.xlsx")
if  ticker == "MCB":
   data=pd.read_excel("DATA SET/MCB_merged_final.xlsx")
if  ticker == "MEBL":
   data=pd.read_excel("DATA SET/MEBL_merged_final.xlsx")
if  ticker == "MTL":
   data=pd.read_excel("DATA SET/MTL_merged_final.xlsx")
if  ticker == "MUGHAL":
   data=pd.read_excel("DATA SET/MUGHAL_merged_final.xlsx")
if  ticker == "MUREB":
   data=pd.read_excel("DATA SET/MUREB_merged_final.xlsx")
if  ticker == "NATF":
   data=pd.read_excel("DATA SET/NATF_merged_final.xlsx")
if  ticker == "NBP":
   data=pd.read_excel("DATA SET/NBP_merged_final.xlsx")
if  ticker == "NESTLE":
   data=pd.read_excel("DATA SET/NESTLE_merged_final.xlsx")
if  ticker == "NML":
   data=pd.read_excel("DATA SET/NML_merged_final.xlsx")
if  ticker == "NRL":
   data=pd.read_excel("DATA SET/NRL_merged_final.xlsx")
if  ticker == "OGDC":
   data=pd.read_excel("DATA SET/OGDC_merged_final.xlsx")
if  ticker == "PABC":
   data=pd.read_excel("DATA SET/PABC_merged_final.xlsx")
if  ticker == "PAEL":
   data=pd.read_excel("DATA SET/PAEL_merged_final.xlsx")
if  ticker == "PAKT":
   data=pd.read_excel("DATA SET/PAKT_merged_final.xlsx")
if  ticker == "PGLC":
   data=pd.read_excel("DATA SET/PGLC_merged_final.xlsx")
if  ticker == "PIBTL":
   data=pd.read_excel("DATA SET/PIBTL_merged_final(ND).xlsx")
if  ticker == "PIOC":
   data=pd.read_excel("DATA SET/PIOC_merged_final.xlsx")
if  ticker == "POL":
   data=pd.read_excel("DATA SET/POL_merged_final.xlsx")
if  ticker == "POML":
   data=pd.read_excel("DATA SET/POML_merged_final.xlsx")
if  ticker == "PPL":
   data=pd.read_excel("DATA SET/PPL_merged_final.xlsx")
if  ticker == "PSEL":
   data=pd.read_excel("DATA SET/PSEL_merged_final.xlsx")
if  ticker == "PSMC":
   data=pd.read_excel("DATA SET/PSMC_merged_final.xlsx")
if  ticker == "PSO":
   data=pd.read_excel("DATA SET/PSO_merged_final.xlsx")
if  ticker == "PSX":
   data=pd.read_excel("DATA SET/PSX_merged_final.xlsx")
if  ticker == "PTC":
   data=pd.read_excel("DATA SET/PTC_merged_final.xlsx")
if  ticker == "RMPL":
   data=pd.read_excel("DATA SET/RMPL_merged_final(LD).xlsx")
if  ticker == "SCBPL":
   data=pd.read_excel("DATA SET/SCBPL_merged_final.xlsx")
if  ticker == "SEARL":
   data=pd.read_excel("DATA SET/SEARL_merged_final.xlsx")
if  ticker == "SHEL":
   data=pd.read_excel("DATA SET/SHEL_merged_final.xlsx")
if  ticker == "SHFA":
   data=pd.read_excel("DATA SET/SHFA_merged_final.xlsx")
if  ticker == "SNGP":
   data=pd.read_excel("DATA SET/SNGP_merged_final.xlsx")
if  ticker == "SRVI":
   data=pd.read_excel("DATA SET/SRVI_merged_final(LD).xlsx")
if  ticker == "SYS":
   data=pd.read_excel("DATA SET/SYS_merged_final.xlsx")
if  ticker == "TGL":
   data=pd.read_excel("DATA SET/TGL_merged_final.xlsx")
if  ticker == "THALL":
   data=pd.read_excel("DATA SET/THALL_merged_final.xlsx")
if  ticker == "TRG":
   data=pd.read_excel("DATA SET/TRG_merged_final.xlsx")
if  ticker == "UBL":
   data=pd.read_excel("DATA SET/UBL_merged_final.xlsx")
if  ticker == "UNITY":
   data=pd.read_excel("DATA SET/UNITY_merged_final.xlsx")
if  ticker == "UPFL":
   data=pd.read_excel("DATA SET/UPFL_merged_final(LD).xlsx")
if  ticker == "YOUW":
   data=pd.read_excel("DATA SET/YOUW_merged_final.xlsx")
button_clicked = st.button("click here to show data")
if button_clicked:
      st.write(data)
data_info = data.info()
data.head()
# Convert 'Date' column to datetime format if it's not already
data['Date'] = pd.to_datetime(data['Date'])

# Check for missing values and handle them if necessary
data = data.dropna()  # This drops rows with any missing values

# Check data types and basic information
print(data.info())
def convert_volume(value):
    """
    Convert volume string to numeric.
    Example: '253.50K' -> 253500
    """
    if 'K' in value:
        return float(value.replace('K', '')) * 1000
    elif 'M' in value:
        return float(value.replace('M', '')) * 1000000
    else:
        return float(value)

# Convert 'Vol.' column to float after handling string values
data['Vol.'] = data['Vol.'].apply(convert_volume)

# Now check if the 'EPS' and 'K_Offer' columns are in numeric format, if not, convert them as well
data['EPS'] = pd.to_numeric(data['EPS'], errors='coerce')  # errors='coerce' will convert non-convertible values to NaN
data['K_Offer'] = pd.to_numeric(data['K_Offer'], errors='coerce')

# After conversion, check the data again
print(data[['Price', 'EPS', 'K_Offer', 'Open', 'High', 'Low', 'Vol.']].head())
# Ensure the dataset only includes numeric columns for correlation analysis
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = data[numeric_columns].corr()

# Then, print the correlation of these numeric features with 'Price'
print(correlation_matrix['Price'].sort_values(ascending=False))
from sklearn.preprocessing import MinMaxScaler

# Assume you've determined these features as significant
features = ['Price', 'EPS', 'K_Offer', 'High', 'Low']
data_selected = data[features]

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_selected)
import numpy as np

def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data)-sequence_length):
        x = data[i:(i+sequence_length)]
        y = data[i+sequence_length, 0]  # Assuming the 'Price' column is the target variable
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 5  # For example, use last 5 days to predict the next day
X, y = create_sequences(data_scaled, sequence_length)
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, len(features))))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)
from sklearn.metrics import mean_squared_error

# Make predictions
predictions = model.predict(X_test)

# Inverse transform to get actual price
predictions_actual = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], len(features)-1))), axis=1))[:,0]

# Compute RMSE
mse = mean_squared_error(y_test, predictions_actual)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

print(mse)
from statsmodels.tsa.stattools import adfuller

# Assuming 'Price' is the target variable
price_series = data['Price']

# Perform Augmented Dickey-Fuller test
adf_test = adfuller(price_series)
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF and PACF
plot_acf(price_series)
plt.show()

plot_pacf(price_series)
plt.show()

# These plots can help in choosing the values for p and q
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
# Example parameters used here, you should use the ones identified from your analysis
model = ARIMA(price_series, order=(5,1,0))  # Adjust (p,d,q) based on your data
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())
# Forecast
forecast = model_fit.forecast(steps=5)  # For example, forecasting next 5 periods
print(forecast)

# To evaluate you would compare these forecasts against actual observed values using RMSE or another relevant metric
# Assume you have a price_series variable from the 'Price' column
train_size = int(len(price_series) * 0.8)
train, test = price_series[:train_size], price_series[train_size:]
from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model (ensure you choose the appropriate order based on prior analysis)
model = ARIMA(train, order=(5,1,0))  # Example order
model_fit = model.fit()
# Predict
predictions = model_fit.forecast(steps=len(test))

# Calculate evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
rmse = mse ** 0.5

MAE=print(f'MAE: {mae}')
MSE=print(f'MSE: {mse}')
RMSE=print(f'RMSE: {rmse}')
st.info(f"MAE: {mae}")
st.success(f"MSE: {mse}")
st.info(f"RMSE: {rmse}")
import streamlit as st
import plotly.graph_objects as go

# Assuming test_dates represents the corresponding dates for the test dataset
# Assuming predictions_actual contains the LSTM predicted prices
test_dates = [datetime.date(2022, 1, 1), datetime.date(2022, 1, 2), datetime.date(2022, 1, 3)]
# Actual vs. LSTM Predicted Prices
trace1_lstm = go.Scatter(
    y=test,  # Actual prices
    x=test_dates,
    mode='lines',
    name='Actual Price'
)
trace2_lstm = go.Scatter(
    y=predictions_actual,  # LSTM Predicted prices
    x=test_dates,
    mode='lines',
    name='LSTM Predicted Price'
)

# Define layout for LSTM plot
layout_lstm = go.Layout(
    title='Stock Price Prediction with LSTM',
    xaxis={'title': 'Date'},
    yaxis={'title': 'Price'}
)

# Combine traces and layout in a figure for LSTM
fig_lstm = go.Figure(data=[trace1_lstm, trace2_lstm], layout=layout_lstm)
# Assuming predictions contains the ARIMA predicted prices

# Actual vs. ARIMA Predicted Prices
trace1_arima = go.Scatter(
    y = test,  # Actual prices
    x = test_dates,
    mode = 'lines',
    name = 'Actual Price'
)
trace2_arima = go.Scatter(
    y = predictions,  # ARIMA Predicted prices
    x = test_dates,
    mode = 'lines',
    name = 'ARIMA Predicted Price'
)

# Define layout for ARIMA plot
layout_arima = go.Layout(
    xaxis = {'title': 'Date'},
    yaxis = {'title': 'Price'}
)

# Combine traces and layout in a figure for ARIMA
fig_arima = go.Figure(data=[trace1_arima, trace2_arima], layout=layout_arima)

# Show the ARIMA plot
st.plotly_chart(fig_arima)
button_clicked2 = st.button("click here to see full chart")
if button_clicked2:
  st.plotly_chart(fig_arima,use_container_width=True)



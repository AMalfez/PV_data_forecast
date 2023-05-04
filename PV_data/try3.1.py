#!/usr/bin/env python
# coding: utf-8

# In[9]:


from ipynb.fs.full.MEMD_all import memd
import streamlit as st
import pandas as pd
import datetime as dt
from geopy.geocoders import Nominatim
import pvlib
from pvlib import solarposition
import numpy as np
import math
import datetime
import pytz
from astral.sun import sun
from astral import LocationInfo
from sklearn.metrics import mean_squared_error
import math
import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import save_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from pickle import dump, load
from attention import Attention
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model, Model
import tensorflow as tf

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
import warnings

warnings.simplefilter("ignore", UserWarning)

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from datetime import time

cc1 = time.fromisoformat("06:55:00")
cc12 = time.fromisoformat("17:05:00")


def split_data(df2, train_size):
    train_days = math.floor(len(df2) * train_size / 44)
    train_data, test_data = df2.iloc[0:train_days * 44], df2.iloc[train_days * 44:len(df2)]
    return train_data, test_data


def create_trainable_dataset(dataframe, n_inputs, n_outputs):
    X, Y = list(), list()
    for i in range(len(dataframe) - n_inputs - n_outputs + 1):
        X.append(dataframe.iloc[i:(i + n_inputs), :])
        Y.append(dataframe.iloc[i + n_inputs:i + n_inputs + n_outputs, -1])
    return np.array(X), np.array(Y)


# Set the title of the app
st.title("Solar PV Forecasting")

# Task 1: Ask the user to input location at which PV plant is located
location = st.text_input("Enter the location of your PV plant:")

# Task 2: Extract the longitude and latitude using geopy

if location:
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(location)
if location is not None:
    lat, lon = location.latitude, location.longitude
    st.write(f"Latitude: {lat:.3f}, Longitude: {lon:.3f}")
else:
    st.write("Unable to get the location coordinates. Please try again.")

# create a dropdown menu for the user to select one of two values
options = ["PV power data only", "PV power + weather data"]
selected_option = st.selectbox("Select an option:", options)

# display the selected option back to the user
st.write("You selected:", selected_option)

# if the user selected "PV power data only", display a sample table with two columns: "Date" and "PV power"
if selected_option == "PV power data only":
    # create a sample dataframe with two columns: "Date" and "PV power"
    dates = pd.date_range("2023-01-01 7:00:00", "2023-01-07", freq="15T")
    pv_power = [2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    data = {"Date": dates[0:7], "PV power": pv_power}
    df = pd.DataFrame(data)

    # format the date column and the PV power column
    df["Date"] = df["Date"].dt.strftime("%d-%m-%Y %H:%M:%S")
    df["PV power"] = df["PV power"].astype(str) + " kW"

    # display the sample table to the user
    st.write("Sample table:")
    st.write(df)

# if the user selected "PV power + weather data", display a sample table with multiple columns: "Date", "Ambient temperature", "Irradiation", "Wind Speed", "Relative Humidity", "Cloud Cover", and "Rainfall"
if selected_option == "PV power + weather data":
    # create a sample dataframe with multiple columns: "Date", "Ambient temperature", "Irradiation", "Wind Speed", "Relative Humidity", "Cloud Cover", and "Rainfall"
    dates = pd.date_range("2023-01-01 7:00:00", "2023-01-07", freq="15T")
    ambient_temperature = [23.4, 24.5, 25.6, 26.7, 27.8, 28.9, 30.0]
    irradiation = [200, 300, 450, 560, 670, 780, 890]
    wind_speed = [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8]
    relative_humidity = [50, 51, 52, 53, 54, 55, 56]
    cloud_cover = [0, 1, 2, 3, 4, 5, 6]
    rainfall = [0, 0, 0, 0, 0, 0, 0]
    pv_power = [2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9]
    data = {"Date": dates[0:7], "Ambient temperature": ambient_temperature, "Irradiation": irradiation,
            "Wind Speed": wind_speed, "Relative Humidity":
                relative_humidity, "Cloud Cover": cloud_cover, "Rainfall": rainfall, "PV power": pv_power}
    df = pd.DataFrame(data)

    # format the date column and the other columns
    df["Date"] = df["Date"].dt.strftime("%d-%m-%Y %H:%M:%S")
    df["Ambient temperature"] = df["Ambient temperature"].astype(str) + " °C"
    df["Irradiation"] = df["Irradiation"].astype(str) + " W/m²"
    df["Wind Speed"] = df["Wind Speed"].astype(str) + " m/s"
    df["Relative Humidity"] = df["Relative Humidity"].astype(str) + " %"
    df["Cloud Cover"] = df["Cloud Cover"].astype(str) + " oktas"
    df["Rainfall"] = df["Rainfall"].astype(str) + " mm"
    df["PV power"] = df["PV power"].astype(str) + " kW"

    # display the sample table to the user
    st.write(
        "Sample table with variables: Date; Ambient temperature; Irradiation; Wind Speed; Relative Humidity; Cloud Cover; Rainfall; PV power")
    st.write(df)

st.write(
    "Note: Please provide data in the above format only. Strictly follow the variable names and no need to write unit of variables. It is just for your information")
# Task4: Ask the user to uplaod excel or csv file such that it strictly follows the provided sample table (including the variable names)
# Define expected column names
col_names = ['Date', 'PV Power (kW)', 'Solar Irradiation', 'Ambient Temperature', 'Wind Speed', 'Relative Humidity',
             'Cloud Cover', 'Rainfall']

# Ask user to upload file
st.header('Load the Data')
uploaded_file = st.file_uploader('Upload your CSV or Excel file', type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load file into a pandas dataframe
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    # Check if dataframe follows expected format
    if df.columns.tolist() == col_names:
        st.success('File uploaded successfully.')
    else:
        st.error(f'Invalid file format. Expected column names: {", ".join(col_names)}.')

    # Task 5: Calculate Solar Zenith Angle
    # Extract the date and time columns and convert to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
    df = df.set_index('Date')
    # Set the timezone to India Standard Time (IST)
    tz = pytz.timezone('Asia/Kolkata')
    df = df.tz_localize(tz)


    # Calculate solar position
    def get_solar_zenith_angle(latitude, longitude, datetime):
        solar_position = pvlib.solarposition.get_solarposition(datetime, latitude, longitude)
        solar_zenith_angle = solar_position['apparent_zenith']
        return solar_zenith_angle


    df['solar zenith'] = get_solar_zenith_angle(lat, lon, df.index)

    # Extract day number and add it as a column
    # df['day_number'] = df.index.dayofyear

    st.write('Solar Zenith Angle Calculation Done')
    st.write(df)

    from datetime import datetime

    # data processing
    df['time'] = df.index.map(lambda x: datetime.strptime(str(x.time()), '%H:%M:%S').time())
    df = df[(df['time'] > cc1) & (df['time'] < cc12)]

    df2 = df[['Solar Irradiation', 'Ambient Temperature', 'Wind Speed', 'Relative Humidity', 'Cloud Cover', 'Rainfall',
              'solar zenith', 'PV Power (kW)']]

    duplicate = df2[df2.duplicated()]
    df2 = df2.drop_duplicates()

    ###############%%%%%%%%%%%%%%%%%%%%%%########################################
    # Task6: scale/normalize the data
    from sklearn.preprocessing import MinMaxScaler

    # assuming 'data' is a pandas DataFrame containing the input data
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(df2))

    # Task 8: Split data
    n_input = 4  # 1 day 11x1=11 hours
    n_output = 44  # 1 day 11x1=11 hours//per day we have 11 hours
    train_size = 0.8
    train_data, test_data = split_data(df2, train_size)

    # Task 9: Create trainable dataset
    X_train, Y_train = create_trainable_dataset(train_data, n_input, n_output)
    X_test, Y_test = create_trainable_dataset(test_data, n_input, n_output)

    # Task 10: Define models
    # define multivariate LSTM model for pv power + weather data
    model_pvw = Sequential()
    model_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(256, return_sequences=True)(model_input)

    x = Attention(units=8)(x)
    x = Dense(44, activation='relu')(x)
    model_pvw = Model(model_input, x)
    model_pvw.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # define univariate LSTM model for only pv power data
    model_pvo = Sequential()
    model_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(8, return_sequences=True)(model_input)
    x = Attention(units=64)(x)
    x = Dense(8, activation='relu')(x)
    model_pvo = Model(model_input, x)
    model_pvo.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # Task 11: Train the model accordingly
    if selected_option == 'PV power data only':
        model_pvo.fit(X_train, Y_train, epochs=1, batch_size=256, validation_data=(X_test, Y_test),
                      callbacks=[stop_early])
    elif selected_option == 'PV power + weather data':
        model_pvw.fit(X_train, Y_train, epochs=1, batch_size=256, validation_data=(X_test, Y_test),
                      callbacks=[stop_early])
    else:
        print('Invalid selection')

    # Check which model the user selected
    if selected_option == "PV power data only":
        # Use the trained univariate LSTM model to make predictions on the test set
        y_pred = model_pvo.predict(X_test)
        Ypp = pd.DataFrame(Y_test)
        Y2p = pd.DataFrame(y_pred)
        Y_results = pd.DataFrame()
        Y_results['Actual'] = Ypp[0]
        Y_results['Predicted'] = Y2p[0]
        Y_results['Actual'] = (Y_results['Actual'] * [df['PV Power (kW)'].max() - df['PV Power (kW)'].min()]) + df[
            'PV Power (kW)'].min()
        Y_results['Predicted'] = (Y_results['Predicted'] * [df['PV Power (kW)'].max() - df['PV Power (kW)'].min()]) + \
                                 df['PV Power (kW)'].min()
        start_time = pd.Timestamp('2023-02-22 07:00:00')

        time_index = pd.date_range(start=start_time, periods=len(Y_test), freq='15min')

        Y_results['Time'] = time_index
        # Calculate the root mean squared error (RMSE) between the predicted and actual outputs
        rmse = np.sqrt(mean_squared_error(Y_test[0], y_pred[0]))
        # Plot the predicted and actual outputs
        fig, ax = plt.subplots()
        ax.plot(Y_results['Time'], Y_results['Actual'], label='Actual')
        ax.plot(Y_results['Time'], Y_results['Predicted'], label='Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('PV Power (kW)')
        ax.set_title('Actual vs. Predicted PV Power')
        ax.legend()
        st.pyplot(fig)
    elif selected_option == "PV power + weather data":
        # Use the trained multivariate LSTM model to make predictions on the test set
        y_pred = model_pvw.predict(X_test)
        Ypp = pd.DataFrame(Y_test)
        Y2p = pd.DataFrame(y_pred)
        Y_results = pd.DataFrame()
        Y_results['Actual'] = Ypp[0]
        Y_results['Predicted'] = Y2p[0]
        Y_results['Actual'] = (Y_results['Actual'] * [df['PV Power (kW)'].max() - df['PV Power (kW)'].min()]) + df[
            'PV Power (kW)'].min()
        Y_results['Predicted'] = (Y_results['Predicted'] * [df['PV Power (kW)'].max() - df['PV Power (kW)'].min()]) + \
                                 df['PV Power (kW)'].min()
        start_time = pd.Timestamp('2023-02-22 07:00:00')

        time_index = pd.date_range(start=start_time, periods=len(Y_test), freq='15min')

        Y_results['Time'] = time_index
        # Calculate the root mean squared error (RMSE) between the predicted and actual outputs
        rmse = np.sqrt(mean_squared_error(Y_test[0], y_pred[0]))
        # Plot the predicted and actual outputs
        fig, ax = plt.subplots()
        ax.plot(Y_results['Time'][-195:-77], Y_results['Actual'][-195:-77])  # , label='Actual')
        # ax.plot(Y_results['Time'][-111:-23], Y_results['Predicted'][-111:-23])#, label='Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('PV Power (kW)')
        ax.set_title('Forecast PV Power')
        ax.legend()
        st.pyplot(fig)
    else:
        print("Invalid selection.")

# In[ ]:


# In[ ]:

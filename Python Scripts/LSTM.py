#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Starting from loading the libraries required for analysis
# This was first done step by step with each model and cell but was combined finally to make the file cleaner and shorter  

get_ipython().system('pip install pandas ')
get_ipython().system('pip install xgboost')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[3]:


# Load dataset
file_path = 'Monthly_New.csv'
data = pd.read_csv(file_path)

# first few rows of the dataframe
data.head()


# In[4]:


# Creating a unique identifier for each year (sequence from 1 to 74) - To use it as a feature while building models
# Combining 'Year' and 'Month' columns into a new 'Date' column which displays time-series format
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str), format='%Y-%m')

data['t'] = range(1, len(data) + 1)

print(data.head())
data.info()


# ## LSTM Model Production

# Clearly, LSTM model performed the best with lowest MAPE and RMSE. However, it is not accounting for the seasonality and is accounting for only trend. How to get the correct forecast graph. 

# In[6]:


# Defining featues for production
features = ['TotalPrimaryEnergyConsumption', 'TotalPrimaryEnergyExports', 'TotalPrimaryEnergyImports', 't']
target = ['TotalPrimaryEnergyProduction']

# Initializing the MinMaxScaler
scaler = MinMaxScaler()

# Scaling the selected features
scaled_data = scaler.fit_transform(data[features])

# Converting the scaled data back to a dataframe for convenience
scaled_data_df = pd.DataFrame(scaled_data, columns=features, index=data.index)

# Splitting the data into a training set (1973-2016) and a test set (2017-2022)
train_data = scaled_data_df[(data['Year'] >= 1973) & (data['Year'] <= 2016)]
test_data = scaled_data_df[(data['Year'] >= 2017) & (data['Year'] <= 2022)]

# Reshaping the data to fit the LSTM input shape
X_train = train_data.values.reshape((train_data.shape[0], 1, train_data.shape[1]))
X_test = test_data.values.reshape((test_data.shape[0], 1, test_data.shape[1]))

# Since we're going to perform one-step-ahead prediction, the target is the same as the input shifted by one time step
y_train = train_data.values[1:]
y_test = test_data.values[1:]

# Trimming the last sample of the training and test input data since we don't have a label for it
X_train = X_train[:-1, :, :]
X_test = X_test[:-1, :, :]

# Defining the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model_lstm.add(Dense(y_train.shape[1]))

# Compiling the LSTM model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Training the LSTM model
history_lstm = model_lstm.fit(X_train, y_train, epochs=200, batch_size=42, validation_split=0.2, verbose=1)

# Evaluating the LSTM model on test data
test_loss_lstm = model_lstm.evaluate(X_test, y_test)

# Predictions
y_pred_lstm = model_lstm.predict(X_test)

# Mean Squared Error (MSE) for LSTM model
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

# Mean Absolute Error (MAE) for LSTM model
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

# Root Mean Squared Error (RMSE) for LSTM model
rmse_lstm = np.sqrt(mse_lstm)

# Print the results for LSTM model
print("LSTM Model Results for Production:")
print(f'Mean Squared Error (MSE): {mse_lstm:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm:.2f}')
print(f'Mean Absolute Error (MAE): {mae_lstm:.2f}')


# ## LSTM Model Consumption

# In[7]:


# Defining featues for consumption
features = ['TotalPrimaryEnergyProduction', 'TotalPrimaryEnergyExports', 'TotalPrimaryEnergyImports', 't', 'PrimaryEnergyStockChange']
target = ['TotalPrimaryEnergyConsumption']

# Initializing the MinMaxScaler
scaler = MinMaxScaler()

# Scaling the selected features
scaled_data = scaler.fit_transform(data[features])

# Converting the scaled data back to a dataframe for convenience
scaled_data_df = pd.DataFrame(scaled_data, columns=features, index=data.index)

# Splitting the data into a training set (1973-2016) and a test set (2017-2022)
train_data = scaled_data_df[(data['Year'] >= 1973) & (data['Year'] <= 2016)]
test_data = scaled_data_df[(data['Year'] >= 2017) & (data['Year'] <= 2022)]

# Reshaping the data to fit the LSTM input shape
X_train = train_data.values.reshape((train_data.shape[0], 1, train_data.shape[1]))
X_test = test_data.values.reshape((test_data.shape[0], 1, test_data.shape[1]))

# Since we're going to perform one-step-ahead prediction, the target is the same as the input shifted by one time step
y_train = train_data.values[1:]
y_test = test_data.values[1:]

# Trimming the last sample of the training and test input data since we don't have a label for it
X_train = X_train[:-1, :, :]
X_test = X_test[:-1, :, :]

# Defining the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model_lstm.add(Dense(y_train.shape[1]))

# Compiling the LSTM model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Training the LSTM model
history_lstm = model_lstm.fit(X_train, y_train, epochs=200, batch_size=42, validation_split=0.2, verbose=1)

# Evaluating the LSTM model on test data
test_loss_lstm = model_lstm.evaluate(X_test, y_test)

# Predictions
y_pred_lstm = model_lstm.predict(X_test)


# Mean Squared Error (MSE) for LSTM model
mse_lstm = mean_squared_error(y_test, y_pred_lstm)

# Mean Absolute Error (MAE) for LSTM model
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

# Root Mean Squared Error (RMSE) for LSTM model
rmse_lstm = np.sqrt(mse_lstm)

# Print the results for LSTM model
print("LSTM Model Results for Consumption:")
print(f'Mean Squared Error (MSE): {mse_lstm:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lstm:.2f}')
print(f'Mean Absolute Error (MAE): {mae_lstm:.2f}')


# In[ ]:





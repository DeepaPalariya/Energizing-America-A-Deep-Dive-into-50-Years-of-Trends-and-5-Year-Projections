#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and loading dataset

# In[15]:


# Starting from loading the libraries required for analysis
# This was first done step by step with each model and cell but was combined finally to make the file cleaner and shorter  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 


# In[2]:


# Loading data
file_path = 'Monthly_New.csv'
data = pd.read_csv(file_path)

# First few rows
data.head()


# # Checking for any missing values 

# In[3]:


missing_values = data.isnull().sum()
missing_values_percentage = (data.isnull().sum() / len(data)) * 100

missing_data_info = pd.DataFrame({
    'Missing Values': missing_values, 
    'Percentage': missing_values_percentage
})

missing_data_info


# ## Converting data into time series format

# In[4]:


# Creating a unique identifier for each year (sequence from 1 to 74) - To use it as a feature while building models
# Combining 'Year' and 'Month' columns into a new 'Date' column which displays time-series format
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str), format='%Y-%m')

data['t'] = range(1, len(data) + 1)

print(data.head())
data.info()


# The dataset does not have any missing values, which is excellent for analysis and model training purposes.

# # Normalize data and add features to forecast Production

# In[31]:


# Initializing the MinMaxScaler
scaler = MinMaxScaler()

# Defining features and target variable
features = ['TotalPrimaryEnergyConsumption', 'TotalPrimaryEnergyExports', 'TotalPrimaryEnergyImports', 't', 'PrimaryEnergyStockChange']
target = ['TotalPrimaryEnergyProduction']
scaled_data = scaler.fit_transform(data[features])

# Converting scaled data back to a dataframe
scaled_data_df = pd.DataFrame(scaled_data, columns=features, index=data.index)

# Splitting the data into a training set (1973-2016) and a test set (2017-2022)
train_data = scaled_data_df[(data['Year'] >= 1973) & (data['Year'] <= 2016)]
test_data = scaled_data_df[(data['Year'] >= 2017) & (data['Year'] <= 2022)]

# Reshaping the data to fit the GRU input shape
X_train = train_data.values.reshape((train_data.shape[0], 1, train_data.shape[1]))
X_test = test_data.values.reshape((test_data.shape[0], 1, test_data.shape[1]))

# Since we're going to perform one-step-ahead prediction, the target is the same as the input shifted by one time step
y_train = train_data.values[1:]
y_test = test_data.values[1:]

# Trimming the last sample of the training and test input data coz there is no label for it
X_train = X_train[:-1, :, :]
X_test = X_test[:-1, :, :]

# Confirming the shape
X_train.shape, y_train.shape, X_test.shape, y_test.shape



# In[32]:


# Defining the GRU model
model_gru = Sequential()
model_gru.add(GRU(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model_gru.add(Dense(y_train.shape[1]))
model_gru.compile(optimizer='adam', loss='mean_squared_error')

# Training the GRU model
history_gru = model_gru.fit(X_train, y_train, epochs=200, batch_size=42, validation_split=0.2, verbose=1)

# Evaluating model on test data
test_loss_gru = model_gru.evaluate(X_test, y_test)

# Making Predictions with the GRU model
y_pred_gru = model_gru.predict(X_test)



# ## Error and performance calculation

# In[33]:


# Mean Squared Error (MSE) 
mse_gru = mean_squared_error(y_test, y_pred_gru)

# Mean Absolute Error (MAE) 
mae_gru = mean_absolute_error(y_test, y_pred_gru)

# Root Mean Squared Error (RMSE) 
rmse_gru = np.sqrt(mse_gru)

# Results 
print(f'Mean Squared Error (MSE) for GRU: {mse_gru:.2f}')
print(f'Mean Absolute Error (MAE) for GRU: {mae_gru:.2f}')
print(f'Root Mean Squared Error (RMSE) for GRU: {rmse_gru:.2f}')


# **Analysis**:
# 
# - The MSE and RMSE values indicate that the model is making accurate predictions, with relatively small errors. The MAE value represents the absolute prediction error.
# 
# - Specifically, for the Production test set, the MSE is 0.04, RMSE is 0.21, and MAE is 0.17. These low error values suggest that the GRU model performed well in predicting Total Primary Energy Production.

# # Plotting Actual Vs Forecast Total Primary Energy Production

# In[34]:


y_pred_actual = scaler.inverse_transform(y_pred_gru)
y_test_actual = scaler.inverse_transform(y_test)


plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:, 0], label='Actual TotalPrimaryEnergyProduction')
plt.plot(y_pred_actual[:, 0], label='Forecasted TotalPrimaryEnergyProduction')
plt.title('Total Primary Energy Production (GRU Model) - Actual Vs Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Original Value')
plt.legend()
plt.show()


# **Analysis**:
# 
# - The visual comparison between the actual and forecasted plots is promising, indicating that the predictions have low errors and the forecasts are accurate.
# 

# # Normalize data and add features to forecast Consumption

# In[35]:


scaler = MinMaxScaler()

features = ['TotalPrimaryEnergyProduction', 'TotalPrimaryEnergyExports', 'TotalPrimaryEnergyImports','t', 'PrimaryEnergyStockChange']
target = ['TotalPrimaryEnergyConsumption']

scaled_features = scaler.fit_transform(data[features])
scaled_target = scaler.fit_transform(data[target])

scaled_data_df = pd.DataFrame(scaled_features, columns=features, index=data.index)
scaled_target_df = pd.DataFrame(scaled_target, columns=target, index=data.index)

train_data = scaled_data_df[(data['Year'] >= 1973) & (data['Year'] <= 2016)]
test_data = scaled_data_df[(data['Year'] >= 2017) & (data['Year'] <= 2022)]

y_train = scaled_target_df[(data['Year'] >= 1973) & (data['Year'] <= 2016)]
y_test = scaled_target_df[(data['Year'] >= 2017) & (data['Year'] <= 2022)]

X_train = train_data.values.reshape((train_data.shape[0], 1, train_data.shape[1]))
X_test = test_data.values.reshape((test_data.shape[0], 1, test_data.shape[1]))


X_train = X_train[:-1, :, :]
y_train = y_train[:-1]
X_test = X_test[:-1, :, :]
y_test = y_test[:-1]


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[36]:


# Defining the GRU model
model_gru = Sequential()
model_gru.add(GRU(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model_gru.add(Dense(y_train.shape[1]))
model_gru.compile(optimizer='adam', loss='mean_squared_error')

# Training the GRU model
history_gru = model_gru.fit(X_train, y_train, epochs=200, batch_size=42, validation_split=0.2, verbose=1)

# Evaluating model on test data
test_loss_gru = model_gru.evaluate(X_test, y_test)

# Making Predictions with the GRU model
y_pred_gru = model_gru.predict(X_test)



# In[37]:


# Mean Squared Error (MSE) 
mse_gru = mean_squared_error(y_test, y_pred_gru)

# Mean Absolute Error (MAE) 
mae_gru = mean_absolute_error(y_test, y_pred_gru)

# Root Mean Squared Error (RMSE) 
rmse_gru = np.sqrt(mse_gru)

# Results 
print(f'Mean Squared Error (MSE) for GRU: {mse_gru:.2f}')
print(f'Mean Absolute Error (MAE) for GRU: {mae_gru:.2f}')
print(f'Root Mean Squared Error (RMSE) for GRU: {rmse_gru:.2f}')


# **Analysis**:
# 
# - For the Consumption test set, the MSE is 0.00, RMSE is 0.05, and MAE is 0.04. These low error values suggest that the GRU model performed exceptionally well in predicting Total Primary Energy Consumption.
# 
# - In summary, the GRU model demonstrated good performance for both Production and Consumption forecasts.

# In[38]:


y_pred_actual = scaler.inverse_transform(y_pred_gru)
y_test_actual = scaler.inverse_transform(y_test)


plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:, 0], label='Actual TotalPrimaryEnergyConsumption')
plt.plot(y_pred_actual[:, 0], label='Forecasted TotalPrimaryEnergyConsumption')
plt.title('Total Primary Energy Consumption (GRU Model) - Actual Vs Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Original Value')
plt.legend()
plt.show()


# **Analysis**:
# 
# - The visual comparison between the actual and forecasted plots indicates a high level of accuracy, validating that the predictions have low errors, and the forecasts are quite accurate.
# 

# In[ ]:





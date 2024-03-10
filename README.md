# Energizing-America-A-Deep-Dive-into-50-Years-of-Trends-and-5-Year-Projections - Python 
 This project conducts a 50-year retrospective analysis of US energy consumption and production, scrutinizing key variables and generating forecasts for 2022-2017. The focus is on evaluating forecast accuracy against actual data to glean insights into the nation's energy landscape evolution.

# Energy Forecasting Project

## Overview
This project aims to conduct a comprehensive analysis of the historical trends in energy consumption and production in the United States, spanning a period of 50 years. Additionally, it seeks to examine the evolving dynamics of various variables such as Total Primary Energy Exports and Imports, Total Primary Energy Consumed by the Industrial and Residential Sectors, and their impact on overall Total Primary Energy consumption and production. The project involves the formulation of forecasts for total primary energy consumption and production for the five-year period (2022-2017) and the models will be fitted for the rest of the years. The ultimate goal is to assess the accuracy of the forecasts by comparing them with the actual values available in the dataset.

## Files

### 1. README.md
   - **Description:** This document serves as a guide to understanding the project. It includes essential information about the project's purpose, the structure of files, and instructions on how to read and test the project.
   - **Contents:**
      - **Project Overview:** This project is a comprehensive exploration of energy dynamics in the United States over the past 50 years, with a primary focus on consumption and production trends. The analysis extends beyond mere observation, scrutinizing the complex relationships among variables such as Total Primary Energy Exports and Imports, Total Primary Energy Consumption by the Industrial and Residential Sectors, and their collective impact on overall energy patterns. The project's core objective is to generate robust forecasts for Total Primary Energy Consumption and Production over a five-year (2022-2017) horizon. The forecasting models will be meticulously crafted and fine-tuned for subsequent validation against actual data. The ultimate goal is to evaluate the accuracy of these forecasts based on different models.
      - **Files Included:** Jupyter Notebooks, Python scripts, CSV files with cleaned data used for analysis, a PowerPoint presentation, HTML files, and Raw data files.
    
### 2. Jupyter Notebooks
   - `DataCleaning_Months.ipynb`
   - `EDA_Months.ipynb`
   - `Winters Exponential.ipynb`
   - `ARIMA and SARIMA.ipynb`
   - `RNN and XGBoost.ipynb`
   - **Description:** Each notebook corresponds to a specific stage in the project, containing code, analysis, and visualizations.

### 3. Scripts
   - `DataCleaning_Months.py`
   - `EDA_Months.py`
   - `Winters Exponential.py`
   - `ARIMA and SARIMA.py`
   - `RNN and XGBoost.py`
   - **Description:** Python scripts corresponding to each model or analysis. These scripts may be used independently of Jupyter notebooks.

### 4. Data Files
   - `Total.txt`
   - `TotalEnergyRaw.csv`
   - `MonthlyData.csv`
   - `Monthly_New.csv`
   - **Description:** "Total.txt" file is the actual bulk upload from the data source. "TotalEnergyRaw" file is a CSV version of Total.txt file. "MonthlyData" is the extracted data after cleansing done in Python. "Monthly_New.csv" is the final cleaned version (where the final reshaping was done in Excel) and is used for the entire analysis.

### 5. Documents
   - A PowerPoint Presentation
   - **Description:** A PowerPoint Presentation that explains everything done in the project.

### 6. How to Run
   - **Instructions:** All the Jupyter notebooks are executable if all the cells are run in order.

### 7. Requirements
   - **Dependencies:** There are no dependencies as long as cells are run in order.

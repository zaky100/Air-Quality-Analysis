# Beijing Air Quality Analysis & Forecasting
Author: [Zakaria Ahmed / 20341433]

 Project Overview
This repository contains a comprehensive data science pipeline analyzing environmental and meteorological data from Beijing monitoring stations. The goal is to understand spatial pollution trends and predict future air quality.
# Key Features
**Data Engineering: Merged fragmented datasets and utilized linear interpolation to mathematically estimate missing sensor readings without disrupting time-series continuity.
* Exploratory Data Analysis (EDA) Visualized the inverse relationship between wind speed and PM2.5, and mapped the drastic seasonal pollution spikes using Seaborn and Matplotlib.
* SARIMA Forecasting: Built a time-series model to recognize the 12-month recurring winter pollution cycle and forecast the next year's trends.
* Predictive Machine Learning: Trained a Random Forest Regressor to predict real-time PM2.5 levels based on live weather inputs.

# libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

def main():
    # Load in the data
    data = pd.read_csv('file_path.csv')

    # Convert the 'ds' column to datetime format for time series analysis  
    data['ds'] = pd.to_datetime(data['ds'],
                                    format = '%Y-%m-%d')  

    # Check if time series is stationary
    stationarity = adfuller(data['y'])

    print(f"ADF Statistic: {stationarity[0]}")
    print(f"P-value: {stationarity[1]}")
    print(f"Critical Values:")
    for key, value in stationarity[4].items():
        print(f"{key}: {value}")

    # Split into training and test set

    test_weeks = 4
    training_set = data.iloc[:-test_weeks, :]
    test_set = data.iloc[-test_weeks:, :]
    print(test_set.tail(1)) 


    # Extract exogeneous variables such as price
    train_exog = training_set[['Price']]
    test_exog = test_set[['Price']]
    print(test_exog.head())

    print(f"test_weeks: {test_weeks}")
    print(f"test_exog shape: {test_exog.shape}")

    # Initizialize SARIMAX model. Use seasonality parameters of your choice.
    model = auto_arima(y=training_set['y'],
                        X=train_exog,
                        m=52,
                        seasonal=True,
                        stepwise=True)

    print(model.summary())
    
    # Generate predictions
    predictions_sarimax = pd.Series(model.predict(n_periods=test_weeks, X=test_exog),
                                    index=test_set.index).rename('SARIMAX')
       
    print(predictions_sarimax)

    # Visualization of training, test, and forecasted values
    training_set['ds'] = pd.to_datetime(training_set['ds'])
    predictions_sarimax.index = test_set['ds']

    print(predictions_sarimax.head())

    # Set start date for plotting
    start_date = pd.to_datetime('2024-01-01')
    plt.figure(figsize=(12,6))

    training_2024 = training_set[training_set['ds'] >= start_date]
    plt.plot(training_2024['ds'], training_2024['y'], label='Training set')

    plt.plot(test_set['ds'], test_set['y'], label='Test set')

    predictions_2024 = predictions_sarimax[predictions_sarimax.index >= start_date]
    plt.plot(predictions_2024.index, predictions_2024, label='SARIMAX Forecasts')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('SARIMAX Forecasts Model')
    plt.show() 
  

    # MAE AND RMSE -- Model Evaluation
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print(round(mean_absolute_error(test_set['y'], predictions_sarimax),0))
    print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_sarimax)), 0))

    # MAPE FUNCTION
    def MAPE(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape_value = MAPE(test_set['y'], predictions_sarimax)
    print(f"MAPE: {mape_value:.2f}%")

    # Save forecasts to new file
    predictions_sarimax.to_csv('predictions_sarimax.csv', index = True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
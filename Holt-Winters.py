import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    # Load in the data
    data = pd.read_csv('file_path.csv')

    # Convert the 'ds' column to datetime format for time series analysis 
    data['ds'] = pd.to_datetime(data['ds'],
                                 format = '%Y-%m-%d')   

   # Split data into test and training set. 
    test_weeks = 4 # Specify how many weeks you want to use for your test set 
    training_set = data.iloc[:-test_weeks, :]
    test_set = data.iloc[-test_weeks:, :]

    # Forecasting model (Exponential Smoothing - Holt-Winters)
    model = ExponentialSmoothing(endog=training_set['y'], # Replace 'y' with your target column if different
                                trend='mul',
                                seasonal='mul',
                                seasonal_periods=52).fit()

    # Predicting for the test set
    test_set.set_index('ds', inplace=True) # ensure 'ds' is the index for test set
    forecast_start = test_set.index[0]
    forecast_end = test_set.index[-1]
    predictions_hw = model.predict(start=forecast_start, end=forecast_end).rename('HW')


    # Convert 'ds' column to datetime in the training set for time series analysis
    training_set['ds'] = pd.to_datetime(training_set['ds'])

    # Set the index of the forecast predictions to match the test set's 'ds' column
    predictions_hw.index = test_set['ds']

    print(predictions_hw.head()) # Print the first few predictions 

    start_date = pd.to_datetime('2022-01-01') # Update with start date of your choice

    plt.figure(figsize=(12,6))

    # Plot training data, test data, and predictions

    training_2024 = training_set[training_set['ds'] >= start_date]
    plt.plot(training_2024['ds'], training_2024['y'], label='Training Set')

    plt.plot(test_set['ds'], test_set['y'], label='Test Set')

    predictions_2024 = predictions_hw[predictions_hw.index >= start_date]
    plt.plot(predictions_2024.index, predictions_2024, label='Holt-Winters Model Forecasts')

    plt.legend()
    plt.xlabel('Dates')
    plt.ylabel('Amount (dollar)')
    plt.title('Holt-Winters Model Predictions')
    plt.show() 

    plt.show()

    # MAE and RMSE calculation    
    print(round(mean_absolute_error(test_set['y'], predictions_hw),0))
    print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_hw)), 0))

    # MAPE FUNCTION
    def MAPE(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape_value = MAPE(test_set['y'], predictions_hw)
    print(f"MAPE: {mape_value:.2f}%")

    # Save predictions
    predictions_hw.to_csv('predictions_hw.csv', index = True)

if __name__ == '__main__':
    main()
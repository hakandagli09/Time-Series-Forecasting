# libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tbats import TBATS
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    # Load data
    data = pd.read_csv('file_path.csv')

    # Data transformation for time-sereis use
    data['ds'] = pd.to_datetime(data['ds'],
                                 format = '%Y-%m-%d')   
    
    plt.ticklabel_format(style='plain')

    test_weeks = 6
    training_set = data.iloc[:-test_weeks, :]
    test_set = data.iloc[-test_weeks:, :]

    # TBATS model 
    model = TBATS(use_trend=True,
                seasonal_periods=[52]).fit(training_set['y'])

    # Predictions 
    predictions_tbats = pd.Series(model.forecast(steps = len(test_set))).rename('TBATS')
    predictions_tbats.index = test_set.index
    print(predictions_tbats.head())

    # Forecasting and Viz 
    training_set['ds'] = pd.to_datetime(training_set['ds'])
    predictions_tbats.index = test_set['ds']

    print(predictions_tbats.head())

    # Set the start date for plotting
    start_date = pd.to_datetime('2024-01-01')

    plt.figure(figsize=(12,6))

    # Plot training data from 2024 onwards
    training_2024 = training_set[training_set['ds'] >= start_date]
    plt.plot(training_2024['ds'], training_2024['y'], label='Training Set')

    # Plot test data (assuming it's all in 2024 or later)
    plt.plot(test_set['ds'], test_set['y'], label='Testing Set')

    # Plot predictions (assuming they're all in 2024 or later)
    predictions_2024 = predictions_tbats[predictions_tbats.index >= start_date]
    plt.plot(predictions_2024.index, predictions_2024, label='TBATS Forecasts')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('TBATS Model Forecasts')
    plt.show() 

    # MAE AND RMSE -- Model Evaluation
    print(round(mean_absolute_error(test_set['y'], predictions_tbats),0))
    print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_tbats)), 0))

    # MAPE FUNCTION
    def MAPE(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape_value = MAPE(test_set['y'], predictions_tbats)
    print(f"MAPE: {mape_value:.2f}%")

    predictions_tbats.to_csv('predictions_tbats.csv', index = True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
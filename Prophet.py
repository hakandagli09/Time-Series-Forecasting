# import libraries 
from prophet import Prophet
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import holidays


def main():
    # Load in the data
    data = pd.read_csv('file_path.csv')

    # Convert the 'ds' column to datetime format for time series analysis  
    data['ds'] = pd.to_datetime(data['ds'],
                                 format = '%Y-%m-%d') 

    # Add a column for holidays (Initially set to 0)
    data['holiday'] = 0

    holidays_dates = holidays

    # Update 'holiday' column in `data` to mark dates that are holidays
    for holiday in holidays_dates:
        data.loc[data['ds'] == holiday, 'holiday'] = 1

    # Create a dataframe that you will then pass to Prophet model
    holidays = pd.DataFrame({'holiday':'holi',
                         'ds': pd.to_datetime(holidays_dates),
                         'lower_window' : -7,
                         'upper_window': 3})
    
    # Remove holiday column, as it is now in holidays df
    data = data.drop(columns = ['holiday'])
    

    # Split data into training and test sets
    test_weeks = 4 
    training_set = data.iloc[:-test_weeks, :]
    test_set = data.iloc[-test_weeks:, :]
    
    # Prophet model. (Configure as needed)
    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                holidays=holidays,
                seasonality_mode='multiplicative',
                seasonality_prior_scale=5,
                holidays_prior_scale=5,
                changepoint_prior_scale=0.05)
    
    # Add price as a regressor, assuming it impacts the target variable. 
    # This can be any features that you think might impact the target variable. 
    m.add_regressor('Price')

    # Fit model on training data
    m.fit(training_set)

    # Create df for forecasting
    future = m.make_future_dataframe(periods = len(test_set),
                                     freq = 'W')
    
    # Concatonate with any additional columns such as price
    future = pd.concat([future, data.iloc[:,2:]],
                   axis = 1)
    
    print(future.head())
    
    # Now generate the forecasts
    forecast = m.predict(future)

    predictions_prophet = forecast.yhat[-test_weeks:].rename('prophet')

    # Visualization 
    predictions_prophet.index = test_set['ds']
    m.plot(forecast)
    m.plot_components(forecast)

    plt.show()

    #MAE and RMSE for Model Evaluation
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print(round(mean_absolute_error(test_set['y'], predictions_prophet),0))
    print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_prophet)), 0))
        
    #MAPE function 
    def MAPE(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(MAPE(test_set['y'], predictions_prophet))
    
    predictions_prophet.to_csv('predictions_prophet.csv', index = True)

if __name__ == '__main__':
    main()
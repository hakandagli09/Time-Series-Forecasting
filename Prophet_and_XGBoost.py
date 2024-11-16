# import libraries 
from prophet import Prophet
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import holidays
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objs as go



def main():
    # Load in the data
    data = pd.read_csv('file_path.csv')

    # Convert the 'ds' column to datetime format for time series analysis 
    data['ds'] = pd.to_datetime(data['ds'],
                                 format = '%Y-%m-%d') 

    # Add a column for holidays (Initially set to 0)
    data['holiday'] = 0

    # Generate the list of US holidays for the years 2022 and 2023
    holidays_dates = holidays.US(years=[2022,2023]) 

    # Create the holidays DataFrame for Prophet (defining the holiday window)
    holidays = pd.DataFrame({'holiday':'holi',
                         'ds': pd.to_datetime(holidays_dates),
                         'lower_window' : -7,
                         'upper_window': 3})
    
    # Add the holiday effects to the dataset    
    data['holiday'] = 0 # reset holiday column
    for holiday in holidays:
        data.loc[data['ds'] == holiday, 'holiday'] = 1
    
    # Remove the 'holiday' column after it's added to Prophet (for clean data) 
    data = data.drop(columns = ['holiday'])
    

    # Prepare training and test sets
    test_weeks = 4
    training_set = data.iloc[:-test_weeks, :]
    test_set = data.iloc[-test_weeks:, :]
    
    # Initialize Prophet model with specified seasonalities and holiday effects
    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                holidays=holidays,
                seasonality_mode='multiplicative',
                seasonality_prior_scale=5,
                holidays_prior_scale=5,
                changepoint_prior_scale=0.05)
    
    # Add 'Price' as a regressor (or any other feature that might help influence the target variable 'y')
    m.add_regressor('Price')

    m.fit(training_set)

    # Create a future dataframe that includes the test period (4 weeks)
    future = m.make_future_dataframe(periods = len(test_set),
                                     freq = 'W')
    
    # Merge regressors 
    future = pd.concat([future, data.iloc[:,2:]],
                   axis = 1)
    
    # Make predictions using trained model
    forecast = m.predict(future)

    # Prophet forecast output with relevant columns (ds, trend, holi, yearly)
    prophet_variables = forecast.loc[:, ['ds', 'trend', 'holi', 'yearly']]

    # Merge prophet variables with original data based on 'ds' column
    df_xgb = pd.merge(data, prophet_variables, on='ds', how='left')

    print(df_xgb.head())  # Check if merge was successful

    # Split XGBoost into training and test set 
    training_set = df_xgb.iloc[:-test_weeks, :]
    test_set = df_xgb.iloc[-test_weeks:, :]

    # Drop unnecesary columns
    training_set = training_set.drop('index', axis=1)
    test_set = test_set.drop('index', axis=1)

    # Isolate X and Y variables 
    y_train = training_set.y
    y_test = test_set.y
    X_train = training_set.iloc[:, 2:]
    X_test = test_set.iloc[:, 2:]

    # Create XGBoost Matrices 
    Train = xgb.DMatrix(X_train, label = y_train)
    Test = xgb.DMatrix(X_test, label = y_test)


    # Set the parameters
    parameters = {'learning_rate': 0.01, 
                  'max_depth' : 5, 
                  'colsample_bytree' : 1,
                  'min_child_weight' : 1,
                  'gamma' : 0, 
                  'random_state' : 1502,
                  'eval_metric' : 'rmse',
                  'objective' : 'reg:squarederror'} 
    
    # Train XGBoost Model 
    model = xgb.train(params = parameters,
                      dtrain = Train,
                      num_boost_round = 100,
                      evals = [[Test, 'y']],
                      verbose_eval = 15)
    
    # Predictions on test set using trained model
    predictions_xgb = pd.Series(model.predict(Test), name = 'XGBoost')
    predictions_xgb.index = test_set['ds'] 
    print(predictions_xgb)

    # Convert back to datetime format
    training_set['ds'] = pd.to_datetime(training_set['ds'])
    test_set['ds'] = pd.to_datetime(test_set['ds'])
    predictions_xgb.index = pd.to_datetime(predictions_xgb.index)

    # Plot the training, test, and forecast data
    start_date = pd.to_datetime('2022-01-01')
    plt.figure(figsize=(12,6))

    training_2024 = training_set[training_set['ds'] >= start_date]
    plt.plot(training_2024['ds'], training_2024['y'], label='Training Amount')

    plt.plot(test_set['ds'], test_set['y'], label='Test Amount')

    predictions_2024 = predictions_xgb[predictions_xgb.index >= start_date]
    plt.plot(predictions_2024.index, predictions_2024, label='XGBoost Model')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.title('XGBoost Forecasts')
    plt.show() 

    # Plotting function:

    def forecast_future(m, model, last_date, end_date):
        periods = pd.date_range(start=last_date + pd.Timedelta(days=7), end=end_date, freq='W').shape[0]

        # Create future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=periods, freq='W')
        future_df = pd.DataFrame({'ds': future_dates})

        # Generate Prophet forecast
        prophet_forecast = m.predict(future_df)
        prophet_features = prophet_forecast[['ds', 'trend', 'yearly', 'holi']]


        # Add Price column (using last known price)
        future_df['Price'] = data['Price'].iloc[-1]

        # Combine all features
        future_features = pd.concat([prophet_features.set_index('ds'), future_df.set_index('ds')], axis=1)
        future_features = future_features[['Price','trend', 'holi', 'yearly']]

        xgb_matrix = xgb.DMatrix(future_features)
        predictions = pd.Series(model.predict(xgb_matrix), index=future_dates, name='Forecast')

        return predictions
    
    last_date = data['ds'].max()
    end_date = pd.to_datetime('2027-12-31')

    future_forecast = forecast_future(m, model, last_date, end_date)
    print(future_forecast)

    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Historic Data'))
    fig_future.add_trace(go.Scatter(x=future_forecast.index, y=future_forecast, mode='lines', name='Future Forecasts'))

    fig_future.update_layout(title='Sales Forecasts',
                             xaxis_title='Date',
                             yaxis_title='Sales')

    fig_future.show()

    future_forecast.to_csv('2028_forecasts.csv', index=True)


    # Model Evaluation
    print('MAE:')
    print(round(mean_absolute_error(test_set['y'], predictions_xgb),0))
    print('RMSE:')
    print(round(np.sqrt(mean_squared_error(test_set['y'], predictions_xgb)), 0))
        
    #MAPE Calculation
    def MAPE(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print('MAPE:')
    print(MAPE(test_set['y'], predictions_xgb))
    
    # Save predictions
    predictions_xgb.to_csv('predictions_xgb.csv', index = True)

if __name__ == '__main__':
    main()
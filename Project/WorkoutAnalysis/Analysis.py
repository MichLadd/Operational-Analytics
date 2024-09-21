import numpy as np, pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from pmdarima.arima import auto_arima

def load_data():
    # Load the data
    df = pd.read_csv('./WorkoutData/preprocessed_data.csv')

    print("Select a column to analyze:")
    for i, column in enumerate(df.columns[1:]):
        print(f"{i}: {column}")
    
    # Ask the user for the column number of the muscle group to analyze
    col_num = int(input("Inserisci il numero della colonna: "))
    
    # Retrieve the column selected by the user
    ds = df.iloc[:, col_num + 1] # +1 perchè salto la colonna della data
    
    return ds

def forecast_accuracy(forecast, actual): 
    # Calculate RMSE and MAE metrics
    rmse = np.sqrt(mean_squared_error(forecast, actual)) # RMSE stands for Root Mean Squared Error, which is a measure of data dispersion (the root brings the measure back to the same scale as the data)
    mae = mean_absolute_error(forecast, actual) # MAE stands for Mean Absolute Error, which means how much the prediction deviates on average from the actual value
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) # MAPE stands for Mean Absolute Percentage Error, which is a measure of the accuracy of a forecasting method in statistics; to calculate the mean absolute percentage error, the percentage difference between the predicted value and the actual value is calculated and the average of these differences is taken

    print(f"RMSE: {rmse} , MAE: {mae}, MAPE: {mape}")

    return({'mae': mae, 'rmse':rmse, 'mape': mape})

def plot_data(dataset, train_pred_x, train_pred_y, test_pred_x, test_pred_y, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(dataset, label='Plot the results')
    plt.plot(train_pred_x, train_pred_y, label=f'Train predictions {model_name}')
    plt.plot(test_pred_x, test_pred_y, label=f'Test predictions {model_name}')
    plt.title(f'Predictions {model_name}')
    plt.xlabel('Date')
    plt.ylabel('Tonnage')
    plt.legend()
    plt.grid(True)
    plt.show()

### ARIMA

def auto_arima(ds, train_perc, seasonal=True):
    train_size = int(len(ds) * train_perc)
    train, test = ds[:train_size], ds[train_size:]
    model = auto_arima(train, start_p=1, start_q=1, 
                       test='adf', max_p=1, max_q=1, m=52, # m=52 for annual weeks
                       start_P=0, seasonal=seasonal,
                       d=None, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)
    test_predict, confint = model.predict(test.shape[0], return_conf_int=True) 

    train_size = len(train)

    forecast_accuracy(test_predict, test)

    plot_data(dataset=ds, train_pred_x=np.arange(train_size).reshape(-1, 1), train_pred_y=train,
              test_pred_x=np.arange(train_size, train_size + len(test_predict)), test_pred_y=test_predict, model_name='sarima')

### SARIMA Grid Search

def sarima_grid_search(ds, train_perc):
    def sarima_model(ds, order, seasonal_order):
        model = SARIMAX(ds, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        return model_fit
    
    train_size = int(len(ds) * train_perc)
    train, test = ds[:train_size], ds[train_size:]

    '''
    p: Autoregressive order (AR) - number of lags in past data used to predict the current value.
    d: Differencing - number of differences needed to make the series stationary.
    q: Moving average order (MA) - number of past errors used to predict the current value.
    P: Seasonal autoregressive order (SAR) - number of seasonal lags in past data.
    D: Seasonal differencing - number of seasonal differences needed to make the series stationary.
    Q: Seasonal moving average order (SMA) - number of past seasonal errors.
    s: Seasonal period - length of the seasonal cycle (e.g., 12 for monthly data with annual seasonality).
    '''
    orders = [(0, 1, 1), (0, 1, 1), (0, 1, 2),(1, 1, 1),(1, 1, 2), (2, 1, 1), (2, 1, 2)]
    seasonal_orders = [(1, 1, 1, 52), (1, 1, 0, 52), (0, 1, 1, 52), (1, 1, 1, 26), (1, 1, 0, 26), (0, 1, 1, 26)]

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    best_model = None

    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                model_fit = sarima_model(train, order, seasonal_order)
                '''
                Manual Grid Search:
                try different parameter combinations and compare the results using AIC (Akaike Information Criterion)
                '''
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model = model_fit
            except:
                continue

    print(f"Best SARIMA model: Order={best_order}, Seasonal Order={best_seasonal_order}, AIC={best_aic}")
    print(best_model.summary())
    
    train_predict = best_model.predict(start=0, end=train_size-1)
    test_predict = best_model.predict(test.shape[0], return_conf_int=True) 
    
    forecast_accuracy(test_predict, test)

    plot_data(dataset=ds, train_pred_x=np.arange(len(train_predict)).reshape(-1, 1), train_pred_y=train_predict,
              test_pred_x=np.arange(train_size, train_size + len(test_predict)), test_pred_y=test_predict, model_name='Sarima Grid Search')

### MULTILAYER PERCEPTRON MODEL (LSTM)

def sequential_lstm(ds, train_perc, look_back=1):
    # Prepare data for the LSTM model
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    train_size = int(len(ds) * train_perc)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds_scaled = scaler.fit_transform(ds.values.reshape(-1, 1))

    train_scaled = ds_scaled[:train_size]
    test_scaled = ds_scaled[train_size:]

    # Create the datasets for the LSTM model
    trainX, trainY = create_dataset(train_scaled, look_back)
    testX, testY = create_dataset(np.concatenate((train_scaled[-look_back:], test_scaled)), look_back)

   # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1, activation='relu'))  # Use ReLU to ensure non-negative output(we are tolking about kg week tonnage)
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # Predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    # Inverse transform the predictions
    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform([trainY])
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])

    # Plot the results
    train_prd_y = train_predict.flatten() 
    train_prd_x = np.arange(start=look_back, stop=len(train_prd_y) + look_back)
    
    test_prd_y = test_predict.flatten()
    test_predict_start = len(train_predict) + (look_back)
    test_prd_x = np.arange(start=test_predict_start, stop=test_predict_start + len(test_prd_y))

    forecast_accuracy(trainY[0], train_predict[:, 0])

    plot_data(ds, train_prd_x, train_prd_y,
                  test_prd_x, test_prd_y, 'LSTM')

# Main function

def main():
    ds = load_data()
    train_perc = 0.8 # consider asking the user

    while True:
        print("Select the analysis method:")
        print("1: ARIMA (auto-arima)")
        print("2: SARIMA (auto-arima)")
        print("3: Sarima Grid Search")
        print("4: Sequential LSTM")
        method = int(input("Enter the method number (0 to terminate): "))

        if method == 1:
            auto_arima(ds, train_perc, seasonal=False)
        elif method == 2:
            auto_arima(ds, train_perc, seasonal=True)
        elif method == 3:
            sarima_grid_search(ds, train_perc)
        elif method == 4:
            sequential_lstm(ds, train_perc, look_back=52)
        elif method == 0:
            break
        else:
            print("Invalid method. Please select a number from the options.")
        continue

    return

if __name__ == '__main__':
    main()
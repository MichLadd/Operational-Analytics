import numpy as np, pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pmdarima.arima import auto_arima

def load_data():
    # Caricare i dati
    df = pd.read_csv('./WorkoutData/preprocessed_data.csv')

    print("Seleziona una colonna da analizzare:")
    for i, column in enumerate(df.columns[1:]):
        print(f"{i}: {column}")
    
    # Chiedo all'utente il numero della colonna del gruppo muscolare da analizzare
    col_num = int(input("Inserisci il numero della colonna: "))
    
    # Recupero la colonna selezionata dall'utente
    ds = df.iloc[:, col_num + 1] # +1 perchè salto la colonna della data
    
    return ds

def forecast_accuracy(forecast, actual): 
    # Calcolo delle metriche RMSE e MAE
    rmse = np.sqrt(mean_squared_error(forecast, actual)) # RMSE acronimo per Root Mean Squared Error, ovvero la radice quadrata dell'errore quadratico medio, è una misura della dispersione dei dati (la radice mi riporta la misura alla stessa scala dei dati)
    mae = mean_absolute_error(forecast, actual) # MAE acronimo per Mean Absolute Error, ovvero l'errore medio assoluto, significa quanto si discosta in media la previsione dal valore reale
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual)) # MAPE acronimo per Mean Absolute Percentage Error, ovvero l'errore percentuale medio assoluto, è una misura della precisione di un metodo di previsione in statistica, per calcolare l'errore percentuale medio assoluto si calcola la differenza percentuale tra il valore previsto e il valore reale e si fa la media di queste differenze

    return({'mae': mae, 'rmse':rmse, 'mape': mape})

def plot_data(dataset, train_pred_x, train_pred_y, test_pred_x, test_pred_y, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(dataset, label='Dati reali')
    plt.plot(train_pred_x, train_pred_y, label=f'Previsioni Train {model_name}')
    plt.plot(test_pred_x, test_pred_y, label=f'Previsioni Test {model_name}')
    plt.title(f'Previsioni {model_name}')
    plt.xlabel('Data')
    plt.ylabel('Tonnellaggio')
    plt.legend()
    plt.grid(True)
    plt.show()

def auto_arima(ds, train_perc, seasonal=True):
    train_size = int(len(ds) * train_perc)
    train, test = ds[:train_size], ds[train_size:]
    model = auto_arima(train, start_p=1, start_q=1, 
                       test='adf', max_p=1, max_q=1, m=52, # m=52 per settimane annuali
                       start_P=0, seasonal=seasonal,
                       d=None, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)
    test_forecast, confint = model.predict(test.shape[0], return_conf_int=True) 

    train_size = len(train)

    plot_data(dataset=ds, train_pred_x=np.arange(train_size).reshape(-1, 1), train_pred_y=train,
              test_pred_x=np.arange(train_size, train_size + len(test_forecast)), test_pred_y=test_forecast, model_name='sarima')


def sequential_lstm(ds, train_perc, look_back=1):
    # Prepara i dati per il modello LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    train_size = int(len(ds) * train_perc)

    # Normalizza i dati
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds_scaled = scaler.fit_transform(ds.values.reshape(-1, 1))

    train_scaled = ds_scaled[:train_size]
    test_scaled = ds_scaled[train_size:]

    # Crea i dataset per il modello LSTM
    trainX, trainY = create_dataset(train_scaled, look_back)
    testX, testY = create_dataset(np.concatenate((train_scaled[-look_back:], test_scaled)), look_back)

    # Creo il modello LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1, activation='relu')) # aggiungo la funzione di attivazione relu all'ultimo layer, non voglio avere valori negativi
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Addestro il modello
    model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)

    # Previsioni
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    # Inverse transform delle previsioni
    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform([trainY])
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])

    # Plot dei risultati
    train_prd_y = train_predict.flatten() 
    train_prd_x = np.arange(start=look_back, stop=len(train_prd_y) + look_back)
    
    test_prd_y = test_predict.flatten()
    test_predict_start = len(train_predict) + (look_back)
    test_prd_x = np.arange(start=test_predict_start, stop=test_predict_start + len(test_prd_y))
    plot_data(ds, train_prd_x, train_prd_y,
                  test_prd_x, test_prd_y, 'LSTM')

    # Calcolo delle metriche RMSE e MAE
    train_rmse = np.sqrt(mean_squared_error(trainY[0], train_predict[:, 0]))
    train_mae = mean_absolute_error(trainY[0], train_predict[:, 0])
    test_rmse = np.sqrt(mean_squared_error(testY[0], test_predict[:, 0]))
    test_mae = mean_absolute_error(testY[0], test_predict[:, 0])

    # Stampa delle metriche
    print(f"Train RMSE: {train_rmse}")
    print(f"Train MAE: {train_mae}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")

def main():
    ds = load_data()
    train_perc = 0.8 # valutare se chiedere all'utente

    while True:
        print("Seleziona il metodo di analisi:")
        print("1: ARIMA (auto-arima)")
        print("2: SARIMA (auto-arima)")
        print("3: Sequential LSTM")
        method = int(input("Inserisci il numero del metodo (0 per terminare): "))

        if method == 1:
            auto_arima(ds, train_perc, seasonal=False)
        elif method == 2:
            auto_arima(ds, train_perc, seasonal=False)
        elif method == 2:
            sequential_lstm(ds, train_perc, look_back=26)
        elif method == 0:
            break
        else:
            print("Metodo non valido. Per favore seleziona un numero tra quelli proposti.")
        continue

    return

if __name__ == '__main__':
    main()
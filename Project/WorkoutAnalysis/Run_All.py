import Analysis

def main():
    df = Analysis.load_data(1)
    ds = df.iloc[:, 1]
    train_perc = 0.8  

    print("Esecuzione di ARIMA (auto-arima)...")
    Analysis.auto_arima_model(ds.copy(), train_perc, seasonal=False, show_plot=False, save_data=True)

    print("Esecuzione di SARIMA (auto-arima)...")
    Analysis.auto_arima_model(ds.copy(), train_perc, seasonal=True, show_plot=False, save_data=True)

    print("Esecuzione di Sarima Grid Search...")
    Analysis.sarima_grid_search(ds.copy(), train_perc, show_plot=False, save_data=True)
    
    print("Esecuzione di Holt Winter’s Exponential Smoothing (HWES)...")
    Analysis.holtwinters(ds.copy(), train_perc, show_plot=False, save_data=True)
    
    print("Esecuzione di Sequential LSTM...")
    Analysis.sequential_lstm(ds.copy(), train_perc, look_back=52, show_plot=False, save_data=True)

    print("Esecuzione di Random Forest...")
    Analysis.random_forest(df.copy(), train_perc, show_plot=False, save_data=True)
    
    print("Esecuzione di Gradient Boosting...")
    Analysis.gradient_boosting(df.copy(), train_perc, show_plot=False, save_data=True)
    
if __name__ == '__main__':
    main()
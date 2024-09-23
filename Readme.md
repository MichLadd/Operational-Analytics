# Operational Analytics
## Project and Example from the Course

The exam involves presenting a project that incorporates elements introduced during the course. 

I've conducted a comprehensive analysis of a gym workout dataset, focusing on data preprocessing, weekly tonnage aggregation, and applying various time series forecasting and machine learning models to evaluate and predict workout trends.

In the 'Example' folder, there are all the examples I've made based on the course material.
In the 'Project' folder, there is the program as well as a dataset example and a folder with all the reports (plots and accuracy metrics).

The project is designed to analyze historical workout data progression:
- First, the `PreProcess.py` script converts weight, sets, and reps columns to numeric, removes rows with NaN values, calculates the total weight lifted (tonnage), aggregates the data weekly by exercise, pivots the table to have exercises as columns, and saves the preprocessed dataset to a CSV file.
- After that, with `Analysis.py`, it is possible to run different functions for time series forecasting and machine learning analysis on workout data, calculating the forecast accuracy and showing the plotted data.
- With `Run_All.py`, it is possible to run all the forecast methods of `Analysis.py` on a selected muscle group column and produce in the 'AnalysisResults' folder an image for the plot and a text file for both train and test accuracy metrics.

Here are all the accuracy metrics from the run on the first muscle group:

## Gradient Boosting_Test_metrics.txt
 - RMSE: 1490.1160729574108 , MAE: 1235.8309830864937, MAPE: inf
## Gradient Boosting_Train_metrics.txt
 - RMSE: 24.90136844469495 , MAE: 20.151016278956906, MAPE: inf
## Holt Winter’s Exponential Smoothing_Test_metrics.txt
 - RMSE: 1623.7550804084278 , MAE: 1214.5637657256955, MAPE: 286278020.6906952
## Holt Winter’s Exponential Smoothing_Train_metrics.txt
 - RMSE: 736.5674601864371 , MAE: 472.80596127493146, MAPE: 4385064.854767861
## LSTM_Test_metrics.txt
 - RMSE: 641.9784008111827 , MAE: 523.22428809569, MAPE: nan
## LSTM_Train_metrics.txt
 - RMSE: 1337.2930302946627 , MAE: 1111.2505012517795, MAPE: nan
## Random Forest_Test_metrics.txt
 - RMSE: 1224.8929850261875 , MAE: 1035.0343783783785, MAPE: inf
## Random Forest_Train_metrics.txt
 - RMSE: 324.8784028032888 , MAE: 255.6375776454422, MAPE: inf
## Sarima (seasonal=False)_Test_metrics.txt
 - RMSE: 461.1628665936784 , MAE: 374.756860888963, MAPE: nan
## Sarima (seasonal=False)_Train_metrics.txt
 - RMSE: 646.4711243094675 , MAE: 509.89752996458105, MAPE: inf
## Sarima (seasonal=True)_Test_metrics.txt
 - RMSE: 595.134889357341 , MAE: 493.5008610185841, MAPE: nan
## Sarima (seasonal=True)_Train_metrics.txt
 - RMSE: 771.9440170036091 , MAE: 586.6534519710009, MAPE: inf
## Sarima Grid Search_Test_metrics.txt
 - RMSE: 1357.9569960944302 , MAE: 1068.4162064011687, MAPE: inf
## Sarima Grid Search_Train_metrics.txt
 - RMSE: 927.1178975142936 , MAE: 751.3795030367468, MAPE: inf
## TBATS_Test_metrics.txt
 - RMSE: 1207.950059118751 , MAE: 951.6688633608636, MAPE: inf
## TBATS_Train_metrics.txt
 - RMSE: 764.611062000937 , MAE: 617.8208587863354, MAPE: inf

The analysis was conducted on a dataset with limited data, resulting in suboptimal outcomes from the predictive models. However, it is already possible to identify some models that better captured the positive trend in tonnage, likely due to their ability to better identify the seasonal patterns of the workouts (cutting and bulking phases).

We can identify the models that performed best on the training set, the test set, and overall (considering both training and test sets).

### Best Models on Training Set
The best models on the training set are those with the lowest RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) values:

## Gradient Boosting:
 - RMSE: 24.90
 - MAE: 20.15

## Random Forest:
 - RMSE: 324.88
 - MAE: 255.64

### Best Models on Test Set
The best models on the test set are those with the lowest RMSE and MAE values:

## TBATS:
 - RMSE: 1207.95
 - MAE: 951.67

## Random Forest:
 - RMSE: 1224.89
 - MAE: 1035.03

### Best Models Overall (Considering Both Training and Test Sets)
To identify the best models overall, we consider the performance on both the training and test sets. The models with consistently low RMSE and MAE values across both sets are:

## Random Forest:
 - Train RMSE: 324.88, Train MAE: 255.64
 - Test RMSE: 1224.89, Test MAE: 1035.03

### Summary
- **Best on Training Set**: Gradient Boosting, Random Forest
- **Best on Test Set**: TBATS, Random Forest
- **Best Overall**: Random Forest, TBATS

These models showed the best performance in terms of RMSE and MAE on the respective datasets.
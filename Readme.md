
# Operational Analytics
## Project and Example from the Course

The exam involves presenting a project that incorporates elements introduced during the course. It is essential to include considerations of data preprocessing, the implementation of at least two predictive algorithms (statistical, neural networks, regression trees), and a statistical comparison of the relative quality of the predictions.
In the 'Example' folder, there are all the examples I've made based on the course material.
In the 'Project' folder, there is the program as well as a dataset example and a folder with all the reports (plots and accuracy metrics).

The project is made to analyze da historical workout data progression:
- First of all the PreProcess.py part converts weight, sets, and reps columns to numeric, removes rows with NaN values, calculates the total weight lifted (tonnage), aggregates the data weekly by exercise, pivots the table to have exercises as columns, and saves the preprocessed dataset to a CSV file.
-  After that, with Analysis.py, is possible to run different functions for time series forecasting and machine learning analysis on workout data, calculating the forecaste accuracy and showing the plotted data.
-  With Runn_All.py is possible to run all the forecast methods of Analysis.py on a selected muscle group column and produce in the 'AnalysisResults' folder an image for the plot and a txt for both train and test accuracy metrics.

Here are all the accuracy metrics made from the run on the first muscle group:

## Gradient Boosting_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1490.1160729574108 , MAE: 1235.8309830864937, MAPE: inf
## Gradient Boosting_Train_metrics.txt (1 corrispondenze)
 - RMSE: 24.90136844469495 , MAE: 20.151016278956906, MAPE: inf
## Holt Winter’s Exponential Smoothing_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1623.7550804084278 , MAE: 1214.5637657256955, MAPE: 286278020.6906952
## Holt Winter’s Exponential Smoothing_Train_metrics.txt (1 corrispondenze)
 - RMSE: 736.5674601864371 , MAE: 472.80596127493146, MAPE: 4385064.854767861
## LSTM_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1797.5510154367585 , MAE: 1353.9038743176673, MAPE: nan
## LSTM_Train_metrics.txt (1 corrispondenze)
 - RMSE: 1407.3105784298468 , MAE: 1076.4538164538164, MAPE: nan
## Random Forest_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1224.8929850261875 , MAE: 1035.0343783783785, MAPE: inf
## Random Forest_Train_metrics.txt (1 corrispondenze)
 - RMSE: 324.8784028032888 , MAE: 255.6375776454422, MAPE: inf
## Sarima (seasonal=False)_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1247.8294028210692 , MAE: 1000.5920554751542, MAPE: inf
## Sarima (seasonal=False)_Train_metrics.txt (1 corrispondenze)
 - RMSE: 803.2092677513716 , MAE: 642.4319616574144, MAPE: nan
## Sarima (seasonal=True)_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1417.736901055506 , MAE: 1100.0858558009602, MAPE: inf
## Sarima (seasonal=True)_Train_metrics.txt (1 corrispondenze)
 - RMSE: 1129.7818213275848 , MAE: 936.6820183245433, MAPE: nan
## Sarima Grid Search_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1357.9569960944302 , MAE: 1068.4162064011687, MAPE: inf
## Sarima Grid Search_Train_metrics.txt (1 corrispondenze)
 - RMSE: 927.1178975142936 , MAE: 751.3795030367468, MAPE: inf
## TBATS_Test_metrics.txt (1 corrispondenze)
 - RMSE: 1207.950059118751 , MAE: 951.6688633608636, MAPE: inf
## TBATS_Train_metrics.txt (1 corrispondenze)
 - RMSE: 764.611062000937 , MAE: 617.8208587863354, MAPE: inf

The analysis was conducted on a dataset with limited data, resulting in suboptimal outcomes from the predictive models. However, it is already possible to identify some models that better captured the positive trend in tonnage, likely due to their ability to better identify the seasonal patterns of the workouts (cutting and bulking phases).

We can identify the models that performed best on the training set, the test set, and overall (considering both training and test sets).

### Best Models on Training Set
The best models on the training set are those with the lowest RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) values:

## Gradient Boosting:

RMSE: 24.90
MAE: 20.15

## Random Forest:

RMSE: 324.88
MAE: 255.64

### Best Models on Test Set
The best models on the test set are those with the lowest RMSE and MAE values:

## TBATS:

RMSE: 1207.95
MAE: 951.67

## Random Forest:

RMSE: 1224.89
MAE: 1035.03

### Best Models Overall (Considering Both Training and Test Sets)
To identify the best models overall, we consider the performance on both the training and test sets. The models with consistently low RMSE and MAE values across both sets are:

## Random Forest:

Train RMSE: 324.88, Train MAE: 255.64
Test RMSE: 1224.89, Test MAE: 1035.03

## TBATS:

Train RMSE: 764.61, Train MAE: 617.82
Test RMSE: 1207.95, Test MAE: 951.67

### Summary
Best on Training Set: Gradient Boosting, Random Forest
Best on Test Set: TBATS, Random Forest
Best Overall: Random Forest, TBATS
These models showed the best performance in terms of RMSE and MAE on the respective datasets.
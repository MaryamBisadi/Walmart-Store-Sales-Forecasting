# Walmart-Store-Sales-Forecasting

This experiment is for predication of weekly sales for each department in 45 Walmart stores located in different regions. Selected holiday markdown events are included in the dataset. These markdowns are known to affect sales.

An exploratory analysis carry out in this project. Both linear and non linear regression predictive model analysis to forecast which departments are affected with the holiday markdown events and the extent of impact.

The regression models that has been analyzed in this experiment includes: linear, Lasso, Ridge, ElasticNet and Random Forest 

The evaluation of each model has been done based on its mean squared error. Regarding to the obtained result random forest regression has the least mean squared error for prediction of weakly sales.

To have better understanding of the dataset the following plots have been generated:
Weakly sales of store 1 based on the dates
![Alt text](/Users/marybisadi/Desktop/Figure_1-1.jpg?raw=true "Store 1 weakly sales")

Sum of weekly sales of different departments for store 1 based on date
![Alt text](/Users/marybisadi/Desktop/Figure_1-2.jpg?raw=true "Sum of store 1 different departments' weakly sales")

Performance on the test data sets for linear regression model
![Alt text](/Users/marybisadi/Desktop/Figure_1-3.jpg?raw=true "Linear Regression")

Mean absolute prediction error(MAPE) and mean squared error(MSSE) are calculated to measure accuracy of linear regression model 

Linear Regression Mean Absolute Prediction Error: 0.08592635208443362
Linear Regression Mean Squared Error 27520243852.51742

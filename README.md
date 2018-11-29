# Walmart-Store-Sales-Forecasting

This experiment is for predication of weekly sales for each department in 45 Walmart stores located in different regions. Selected holiday markdown events are included in the dataset. These markdowns are known to affect sales.

An exploratory analysis carry out in this project. Both linear and non linear regression predictive model analysis to forecast which departments are affected with the holiday markdown events and the extent of impact.

The regression models that has been analyzed in this experiment includes: linear, Lasso, Ridge, ElasticNet and Random Forest 

The evaluation of each model has been done based on its mean squared error. Regarding to the obtained result random forest regression has the least mean squared error for prediction of weakly sales.

To have better understanding of the dataset the following plots have been generated:

### Weakly sales of store 1 based on the dates
![figure_1-1](https://user-images.githubusercontent.com/39537957/49208970-20614d80-f36e-11e8-8d6e-f624599b1b05.png)

### Sum of weekly sales of different departments for store 1 based on date
![figure_1-1](https://user-images.githubusercontent.com/39537957/49208970-20614d80-f36e-11e8-8d6e-f624599b1b05.png)

### Performance on the test data sets for linear regression model
![figure_1-2](https://user-images.githubusercontent.com/39537957/49209111-70401480-f36e-11e8-9f27-7e48ba75fc29.png)

Mean absolute prediction error(MAPE) and mean squared error(MSSE) are calculated to measure accuracy of linear regression model 

Linear Regression Mean Absolute Prediction Error: 0.08592635208443362

Linear Regression Mean Squared Error: 27520243852.51742

The following diagram shows Lasso Regression. It has been used for Dimension Reduction since Lasso penalizing extra features. To find the best penalty parameter mean squared error is calculated for a range of values for penalty parameter.

### Lasso Mean Squared Error for a range of penalty parameters
![figure_1-3](https://user-images.githubusercontent.com/39537957/49209126-7df59a00-f36e-11e8-8879-7a2a9104e807.png)

The same analyses has been done for Ridge and ElasticNet methods as well.
### Ridge Mean Squared Error for a range of penalty parameters
![figure_1-4](https://user-images.githubusercontent.com/39537957/49210604-5b658000-f372-11e8-879a-540aaac23f3b.png)

### ElasticNet Mean Squared Error for a range of penalty parameters
![figure_1-5](https://user-images.githubusercontent.com/39537957/49210799-de86d600-f372-11e8-9058-a7ced90e43e3.png)

For analyzing a non linear regression method for this experiment RandomForest regression method has been chosen. The following diagram shows mean squared error for different number of tree in the forest

### RandomForest Mean Squared Error for different number of tree in the forest
![figure_1-6](https://user-images.githubusercontent.com/39537957/49210844-f8c0b400-f372-11e8-9424-1fedef855805.png)

Based on this diagram the best number of tree for the random forets is 150 and the calculated error for RandomForest method with this number of trees as the parameter is:

Non Linear Regression Mean Absolute Prediction Error: 0.06514781427233526

Non Linear Regression Mean Squared Error 21526476093.822758


By comparing error values of linear and non linear regression methods we find out that for our dataset random forest which is a nonlinear regression method works best

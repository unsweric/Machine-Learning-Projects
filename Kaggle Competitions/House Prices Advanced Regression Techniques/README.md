This is Getting Started Prediction Competition on Kaggle.It is a regression problem which aimed to predict the house price based on 79 features.   
  
There are 12 notebooks in this file:  
(1)Trial_1:This version uses basic feature engineering,  labelEncoder for categorical feature,random forest as predicting model  and is not using  model selecting pipeline.  
  
(2)Trial_2:This version uses basic feature engineering,  labelEncoder for categorical feature,and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor and RandomForestRegressor.And random forest wins! 
  
(3)Trial_3_1-This version uses basic feature engineering, labelEncoder for categorical feature and Neural Network(tensorflow) as predicting model.Apply 7 different parameter combinations to tune the neural nets.  
  
(4)Trial-3-2-This version uses basic feature engineering, labelEncoder for categorical feature and Neural Network(keras and tensorflow) as predicting model.Apply 4 different parameter combinations to tune the keras neural nets and compare the result to the one using tensorflow.  
  
(5)Trial-3-3-This version uses advanced feature engineering(learned from other kernel on kaggle:https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset) and Neural Network(keras) as predicting model.Apply 8 different parameter combinations to tune the Keras neural nets and compare compare the result to the one using Lasso and RandomForest. 
  
(6)Trial_4:This version uses basic feature engineering,  labelEncoder for categorical feature,SelectFromModel to select features and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor and RandomForestRegressor.And random forest wins! 
  
(7)Trial_5:This version uses basic feature engineering,  labelEncoder for categorical feature,PCA for feature reduction  and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor and RandomForestRegressor.And random forest wins! 
  
(8)Trial_6:This version uses basic feature engineering, labelEncoder for categorical feature,ICA for feature reduction and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor and RandomForestRegressor.And random forest wins!  
  
(9)Trial_7:This version uses basic feature engineering, one-hot encode for categorical feature,ICA for feature reduction and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor and RandomForestRegressor.And random forest wins!  
  
(10)Trial_8:This version uses basic feature engineering, one-hot encode for categorical feature,PCA for feature reduction and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor and RandomForestRegressor.And random forest wins!  
  
(11)Trial_9:This version uses basic feature engineering,analyze dataset by pivoting features ,labelEncoder for categorical feature and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor and RandomForestRegressor.And random forest wins!  
  
(12)Trial_10:This version uses advanced feature engineering,varies plotting analysis, and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor,ElasticNet,Lasso and RandomForestRegressor.   
  
  

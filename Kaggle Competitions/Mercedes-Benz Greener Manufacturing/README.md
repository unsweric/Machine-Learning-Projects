This is a awarded competition(total reward $25,000)on kaggle.It is a regression problem,with a dataset representing different permutations of Mercedes-Benz car features to predict the time it takes to pass testing.I ended up ranking 11% of this competition,used XGboosting,emsemble and stacking algorithems.  
  
There are 8 notebooks in this file:  
(1)Trial_1:This version is the benchmark model,uses basic feature engineering, labelEncoder for categorical feature,and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor ,RandomForestRegressor,Lasso and ElasticNet.The model Ridge wins and it then has been carefully tuned by using gridsearch.Scored 0.53514 on kaggle lead board.

(2)Trial_2:This version uses basic feature engineering, One-hot encode for categorical feature,and model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor ,RandomForestRegressor,Lasso and ElasticNet.The model Ridge wins and it then has been carefully tuned by using gridsearch.Scored 0.54295 on kaggle lead board.

(3)Trial_3:This version uses basic feature engineering, One-hot encode for categorical feature,PCA for feature reductionand model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor ,RandomForestRegressor,Lasso and ElasticNet.The model Ridge wins and it then has been carefully tuned by using gridsearch.Also tuned Lasso ,ElasticNet and RandomForest. Scored 0.54305 on kaggle lead board.  
  
(4)Trial_4:This version uses basic feature engineering, One-hot encode for categorical feature,ICA for feature reductionand model selecting pipeline to select the best model from Ridge,SVR(kernel='linear'),SVR(kernel='rbf'),AdaBoostRegressor ,RandomForestRegressor,Lasso and ElasticNet.The model Ridge wins and it then has been carefully tuned by using gridsearch.Also tuned Lasso ,ElasticNet and RandomForest. Scored 0.54305 on kaggle lead board.

(5)Trial_5_1:Now I am trying ensemble and stacking.To be specific,first tune stacking estimator and then tune Xgboost,finally combine these two model's results.Most of this skills are learnt from this kernel:https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697.
This version tune 10 differents model,rank them and prepare them to build stacking estimator.Stacking estimator means build a pipeline which contains best combination of all the tuned model,whth in the pipeline, each prediction by one model is added to the features which are used to feed next model. 

(6)Trial_5_2:This version tune the stacking estimator by firstly find the best combanations of weak model to form the pipleline--the best comb is  stacked_pipeline = make_pipeline(  
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),  
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),  
    linear_model.Ridge(alpha=31.99))  
,and secondly using gridsearch to tune parameter for all these three model,in other words,tune this pipeline--the best parameter found are shown above.The score on  public leadboard is 0.56764,which is not the highest eventhough it should give the best the result.However it scored 0.55206 on private leadboard,which ranked top 8%.(however,due to the lack of experience,I did not choose this score to submit,so I ended up ranking top 11% by choosing the score ranked the highest on the public leadboard )

(7)Trial_6:This version tune the Xgbboost Regressor and the combine the results.

(8)Trial_7:This version tune the Neural Network(Keras),however,the result is not as good as stacking.
 
Reference:
https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697

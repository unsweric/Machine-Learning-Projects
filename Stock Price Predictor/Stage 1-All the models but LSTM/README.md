At this stage,I traing all the model but LSTM.  
There 6 steps in stage:  
(1)Step_1:Collecting and clearning data:I download history price(1997-6-30 2017-6-27) of 8 stocks,namely,Australia ASX,Frankfurt DAX,Dow Jones Industrial Average (^DJI),Hong Kong Hang Seng,Nasdaq,Paris CAC40,S&P500 and Tokyo Nikkei-225.Then I clean all the data,deal with missing data.Please look at the notebook for details.  

(2)Step_2:(1)Feature Engineering:creat three kindf of feature(as shown below)--  
             a,Multiple Day Returns: percentage difference of Adjusted Close Price of i-th day  compared to (i-delta)-th day. Example: 3-days Return is the percentage difference of Adjusted Close Price of today compared to the one of 3 days ago.  
             b,Returns Moving Average: average returns on last delta days. Example: 3-days Return is the percentage difference of Adjusted Close Price of today compared to the one of 3 days ago.  
             c,Time Lagged Returns: shift the daily returns n days backwards. Example: if n =  1 todays’ Return becomes yesterdays’ Return.  
          (2) compare the result of Regression and Classification problem,and finally decide to treat it as a classification problem due to its higher performance.

(3)Step_3:(1),build customized Time Series GridSearch Function,the tradional grid search randomly spit the dataset which is not suitable for time series problem as it stricctly requires using old data to predict new data! Thus,I make this time series gridsearch function,it goes in this way:  
    1.Split train set in " number_folds" consecutive time folds.  
    2.Then, in order not lo lose the time information, perform the following steps:  
    3.Train on fold 1 –>  Test on fold 2  
    4.Train on fold 1+2 –>  Test on fold 3  
    5.Train on fold 1+2+3 –>  Test on fold 4  
    .  
    .  
    .  
    n.Train on fold 1+2+3+...+ "number_folds-1" –>  Test on fold "number_folds"   
    n+1.Compute the average of the accuracies of the "number_folds-1" test folds   
    
         (2),Systematically train 7 models,namely,GradientBoostingClassifier,RandomForestClassifier,KNeighborsClassifier,SVC,AdaBoostClassifier,QuadraticDiscriminantAnalysis and  SGDClassifier.  
         (3)Train the stacking model.Turns out that GradientBoostingClassifier outperfomes all other model includ stacking model.
         
(4)Step_4: change the customized Time Series GridSearch Function by using test dataset as validation set(has risk of overfitting),and follow retrain all the model.
GradientBoostingClassifier outperfomes all other model and the best accuracy of 30 days prediction reach 55%.

(5)Step_5: Systematically train XGboositng Classifier by using  customized Time Series GridSearch Function.  

(6)Step_6: Systematically train XGboositng Classifier by using  customized Time Series GridSearch Function with test dataset as validation set(has risk of overfitting). the best accuracy of 30 days prediction is 54.8%.

Customized Feature Selection funtion:this function can be used to tune the parameters of DaysReturns,DaysReturnMovAvg and DaysLags.This funtion has not bees used at this stage as all the model does not perform very well,probably goona to be used for LSTM.

Conclusion:at this stage,I tried all the model but LSTM,the best model is GradientBoostingClassifier which reaches 55% in terms of  accuracy of 30 days prediction.However,this result is not satsifying enough.Therefor,at nest stage,I will try LSTM(Long Short Term Momery) Neural Network which is very suitable for time series problem.

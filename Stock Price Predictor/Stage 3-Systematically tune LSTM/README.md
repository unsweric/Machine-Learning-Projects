At this stage,I systematically tune LSTM.This tuning takes 1 months,rather than using gridsearch methos to fully tune all combination of parameter,I tune the parameter one by one due to the limitation of my cmputer power and time.According to my calculation,it takes decades to fully tune this model.  

This tuning has been divided into 8 steps,for the result of each step's tuning ,please refer to the excel file"Result of Tuning".In the "results of Tuning ",there are 10 indeies used to measure the parameter,decision making need to consider the trade off between there 10 indeies and most important indeies are mean of test_RMSE,mean of accuracy_all and mean of accuracy_30.

Step 1:Customize the function to prepare the data;build benchmark model-Persistence Model;establish best model from stage 2,both stateless and stateful;Tune the epochs from 100 to 5000,and epochs 100wins.

Step 2:tune the combinations of feature,namely:[Open,High,Low,Close,Adjclose,Vloume],[Open,High,Low,Adjclose,Vloume],[Open,High,Low,Close,Vloume],[Open,High,Low,Close],[Open,High,Low],[High,Low,Close],[Open,Close],[Open,High,Low,Adjclose],[Open,High,Low,Close,Adjclose],and  [Open,High,Low,Close] wins.

Step 3: tune steps,i.e.how many historical days's data are used as features,namely:[2,3,5,7,10,13,15,17,20,23,25,27,30].Based on the result,step 1 and stap 15 provides similar results,I choose step 1 for future parameter tuning for better effcient,but consider step 15 at the final tuning step.

Step 4:tune layers and neurons.To be specific,I tuned 1 layer with neurons[2,3,4,5,7,10,15,20,25,30,40,50,70,100], 2 layer with neurons[[2,2],[3,3],[4,4],[5,5],[10,10],[30,30],[50,50],[70,70],[100,100],[2,4],[4,8],[8,16],[16,32],[32,64]], 3 layer with neurons[[2,2,2],[3,3,3],[4,4,4],[5,5,5],[10,10,10],[30,30,30],[50,50,50],[70,70,70]].And 1 layer with 7 neurons wins.  
PS: as this note book is over 25 mb which is the limit of uplaoding size for github,I delete to graphs the results in it,please refer to excel file"Result of Tuning" for tuning results.

Step 5:(1) tune drop outs,drop 0 wins.  
       (2) retune epochs [50,60,70,80,90,100],epochs 100 wins.  
       (3) compare stateless with stateful,stateful wins.  
       (4) compare with 15 steps,15 steps wins.
       
Step 6: Tune Weight Regularization.
        (1)Bias Weight Regularization:[L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]  
        (2)Input Weight Regularization:[L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
        (3)Recurrent Weight Regularization:[L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
        and bias_regularizer=L1L2(l1=0.01, l2=0.0) wins.
        
Step 7:Tune Optimization Algorithm,namely:['adam','sgd','adagrad','adamax','nadam','adadelta'].the results of  'adam','sgd','adagrad' and 'adamax' are similar,thus I will try four of these for my final tuning.

Step 8:This step I mimics a real-world scenario where new adjclose would be available each day and used in the forecasting of the following day.A rolling-forecast scenario will be used, also called walk-forward model validation.Each time step of the test dataset will be walked one at a time. A model will be used to make a forecast for the time step, then the actual expected value from the test set will be taken and made available to the model for the forecast on the next time step.  
an investment plot will be plotted to check the ultimate useness of this model,The investment strategy goes like this: if the predicted AdjClose price is higher than today's then buy this stock at the open price of the next day and sell it at the close price. Otherwise short it. So if the prediction is accuate, this strategy will make profit, otherwise we lose money. 
PS: as this note book is over 25 mb which is the limit of uplaoding size for github,I delete to graphs the results in it,please refer to excel file"Result of Tuning" for tuning results.

Step 8.1:Adagrad Optimization Algorithm with different updata epochs,1 epochs wins.   
Step 8.2:Adam Optimization Algorithm with different updata epochs,1 epochs wins.  
Step 8.3:Sgd Optimization Algorithm with different updata epochs,2 epochs wins.  
Step 8.4:Adamax Optimization Algorithm with different updata epochs,they all perform pretty bad,Adamax is out of picture.  


Final Model:Based on the all tuning resluts,this is the final model I choose,which will be used to build the wedsite and to forcast fuuture adjclose price of S&P500.

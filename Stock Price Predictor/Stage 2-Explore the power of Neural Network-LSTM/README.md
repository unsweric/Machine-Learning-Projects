At this stage,I explore the power of Neural Nets,specifically,LSTM-a Recurrent Neural Network(RNN).Tune the model by intuition rather than systematically.

There are 4 steps at this stage:  
(1)Step_1：only use only use one day's adjclose price as feature to predict next day's adjclose price,compare the results by using LSTM and Other Regression models.
Conclusion:THe performance of LSTM looks ok,but absolutely needs to be tuned.On the other hand,for other models like ridhe,eventhough their R2 scores are very high ,they  failed to predict to future price as they simply use today's adjclose price to predict tommorrow's,same preformance as persistant model which normaly used as benchmark model for LSTM application.  

(2)Step_2:use one day's daily return as feature to predict next day's daily return,the result is very bad.THerefore,I go back to predict adjclose price itself.  

(3)Step_3:use one day's open,close,high,low and volume as features to predict next day's adjclose price.Tried 8 different combinations of parameters,the best result got accuracy 71% for 30 days' movement predictions .Also tried to mimic the realy world practice by using walk-forward model validation and reach 0.6428 for 30 days' movement predictions.For the details of combination of parameters,please see the notebook.   

(4)Step_4:use using 8 index's open,close,high,low and volume as features predict next day's adjclose price.Tried 3 different combinations of parameters and the results are not satifying.

Conclution: at this stage I tune the LSTM model, choose different of features and try  different combinations of parameters.THe best one reachs 71% for 30 days' movement predictions which can be used as a benchmark for next stage-syytematically tune LSTM.

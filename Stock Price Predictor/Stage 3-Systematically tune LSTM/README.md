At this stage,I systematically tune LSTM.This tuning takes 1 months,rather than using gridsearch methos to fully tune all combination of parameter,I tune the parameter one by one due to the limitation of my cmputer power and time.According to my calculation,it takes decades to fully tune this modeol.  

This tuning has been divided into 8 steps,for the result of each step's tuning ,please refer to the excel file"Result of Tuning".  

Step 1:Customize the function to prepare the data;build benchmark model-Persistence Model;establish best model from stage 2,both stateless and stateful;Tune the epochs from 100 to 5000,and epochs 100wins

this is a awarded competition(total reward $15,000)on kaggle.It is a classification and NPL(Natural Langurage Processing) problem which requires automatically classify genetic variations.I am still doing this one and currently ranking 38%.  
  
There are 3 notebooks in this file:  
(1)Trial_1:This version uses TfidfVectorizer and TruncatedSVD for feature engineering,XGBClassifier as predicting model(use 80% dataset to train).The result is not very good,only scored scored 1.82637 on Kaggle leaderboard.

(2)Trial_2:This version uses TfidfVectorizer and TruncatedSVD for feature engineering,XGBClassifier as predicting model(use 100% dataset to train).The result has been significantly improved, scored 0.87016 on Kaggle leaderboard.

(3)Trial_3:This version uses TfidfVectorizer and TruncatedSVD for feature engineering,GradientBoostingClassifier as predicting model(use 100% dataset to train).The result is pretty good, scored 0.70223 on Kaggle leaderboard.


import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

                



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

#usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values


'''Train the xgb model then predict the test data'''

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
# NOTE: Make sure that the class is labeled 'class' in the data file

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

num_boost_rounds = 1250
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''
#ORIGINAL
class StackingEstimator(BaseEstimator, TransformerMixin):
        
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)

stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
# scored  0.56884


#Tune Stacking Estimator_1
reg_1=LassoLarsCV(normalize=True)
reg_1.fit(finaltrainset, y_train)
X_1 = check_array(finaltrainset)
X_transformed_1 = np.copy(X_1)
X_transformed_1 = np.hstack((np.reshape(reg_1.predict(X_1), (-1, 1)), X_transformed_1))

reg_2=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)
reg_2.fit(X_transformed_1, y_train)
X_2 = check_array(X_transformed_1)
X_transformed_2 = np.copy(X_2)
X_transformed_2 = np.hstack((np.reshape(reg_2.predict(X_2), (-1, 1)), X_transformed_2))

reg_3=LassoLarsCV()
reg_3.fit(X_transformed_2, y_train)
results = reg_3.predict(X_transformed_2)
# only scored 0.49138

#Tune Stacking Estimator_2
reg_1=LassoLarsCV(normalize=True)
reg_1.fit(finaltrainset, y_train)
X_1 = check_array(finaltrainset)
X_transformed_1 = np.copy(X_1)
X_transformed_1 = np.hstack((np.reshape(reg_1.predict(X_1), (-1, 1)), X_transformed_1))

reg_2=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)
reg_2.fit(finaltrainset, y_train)
X_2 = check_array(X_transformed_1)
X_transformed_2 = np.copy(X_2)
X_transformed_2 = np.hstack((np.reshape(reg_2.predict(finaltrainset), (-1, 1)), X_transformed_2))

reg_3=LassoLarsCV()
reg_3.fit(X_transformed_2, y_train)
results = reg_3.predict(X_transformed_2)
# only scored 0.49125

#Tune Stacking Estimator_3
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

class StackingEstimator(BaseEstimator, TransformerMixin):
        
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=linear_model.Ridge(alpha=31.3)),
    StackingEstimator(estimator=linear_model.Lasso(alpha=0.0001)),
    StackingEstimator(RandomForestRegressor(n_estimators=120)),
    ElasticNet(alpha=0.01, l1_ratio=0.0001)

)

stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)
#scored  scored 0.55851

'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))


#Tune Stacking Estimator_4
from sklearn import linear_model
from sklearn.linear_model import ElasticNet

class StackingEstimator(BaseEstimator, TransformerMixin):
        
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=linear_model.Ridge(alpha=31.3)),
    StackingEstimator(estimator=linear_model.Lasso(alpha=0.0001)),
    ElasticNet(alpha=0.01, l1_ratio=0.0001)

)

stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56633


#Tune Stacking Estimator_5
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))



stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56677


#Tune Stacking Estimator_6
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=AdaBoostRegressor(n_estimators=30,learning_rate=0.1)),
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))



stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56488

#Tune Stacking Estimator_7
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=svm.SVR(kernel='linear',C=0.3) ),
    StackingEstimator(estimator=AdaBoostRegressor(n_estimators=30,learning_rate=0.1)),
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))



stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56560

#Tune Stacking Estimator_8
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=linear_model.Lasso(alpha=0.0001)),
    StackingEstimator(estimator=svm.SVR(kernel='linear',C=0.3) ),
    StackingEstimator(estimator=AdaBoostRegressor(n_estimators=30,learning_rate=0.1)),
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))



stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56317


#Tune Stacking Estimator_9
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(n_estimators=150)),
    StackingEstimator(estimator=linear_model.Lasso(alpha=0.0001)),
    StackingEstimator(estimator=svm.SVR(kernel='linear',C=0.3) ),
    StackingEstimator(estimator=AdaBoostRegressor(n_estimators=30,learning_rate=0.1)),
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))



stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored   0.55882

#Tune Stacking Estimator_10
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=svm.SVR(kernel='rbf',C=2.0,epsilon=1.0)), 
    StackingEstimator(estimator=RandomForestRegressor(n_estimators=150)),
    StackingEstimator(estimator=linear_model.Lasso(alpha=0.0001)),
    StackingEstimator(estimator=svm.SVR(kernel='linear',C=0.3) ),
    StackingEstimator(estimator=AdaBoostRegressor(n_estimators=30,learning_rate=0.1)),
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))



stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored   0.55882

#Tune Stacking Estimator_11
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


reg=linear_model.Ridge(alpha=109)

reg.fit(finaltrainset, y_train)
results = reg.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,reg.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored   0.56807

#Tune Stacking Estimator_12
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


reg=linear_model.Ridge(alpha=109,max_iter=10)

reg.fit(finaltrainset, y_train)
results = reg.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,reg.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored   0.56807


#Tune Stacking Estimator_13
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


reg=linear_model.Ridge(alpha=109,max_iter=1)

reg.fit(finaltrainset, y_train)
results = reg.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,reg.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored   0.56807

#Tune Stacking Estimator_14
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


reg=linear_model.ElasticNet(alpha=1.0,l1_ratio=0.01)

reg.fit(finaltrainset, y_train)
results = reg.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,reg.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored   0.55830

#Tune Stacking Estimator_15
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


reg=linear_model.Ridge(alpha=31.99)

reg.fit(finaltrainset, y_train)
results = reg.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,reg.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored   0.56798

#Tune Stacking Estimator_16
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

#estimators = [('Elastic_Net', StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1))), 
#              ('LassoLars_CV', StackingEstimator(estimator=LassoLarsCV(normalize=True))),
#              ('Ridge', linear_model.Ridge(alpha=31.99))]
#pipe = Pipeline(estimators)
#
#pipe.fit(finaltrainset, y_train)
#results = pipe.predict(finaltestset)


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))


stacked_pipeline.steps[0]
stacked_pipeline.steps[1]
stacked_pipeline.steps[2]

StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)).get_params().keys()



params = {'stackingestimator-1__estimator__alpha':[0.0001,0.0003,0.0005,0.0008,0.001,0.0013,0.0015,0.0018,0.002],
          'stackingestimator-1__estimator__l1_ratio':[0.01,0.03,0.05,0.08,0.1,0.13,0.15,0.18,0.2],
          'ridge__alpha':[30.0,30.5,31.0,31.5,32.0,32.5,33.0,33.5,34.0]}
scoring_fnc = make_scorer(r2_score)
grid= GridSearchCV(stacked_pipeline,params,scoring_fnc)  

grid = grid.fit(finaltrainset, y_train)
model2=grid.best_estimator_

print(model2.get_params()['stackingestimator-1__estimator__alpha'])
#0.002
print(model2.get_params()['stackingestimator-1__estimator__l1_ratio'])
#0.01
print(model2.get_params()['ridge__alpha'])
#34.0

model2.fit(finaltrainset, y_train)
results = model2.predict(finaltestset)


'''R2 Score on the entire Train data when averaging'''

print('R2 score on train data:')
print(r2_score(y_train,model2.predict(finaltrainset)*0.2855 + model2.predict(dtrain)*0.7145))

'''Average the preditionon test data  of both models then save it on a csv file'''

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56705


#Tune Stacking Estimator_17
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))


params = {'stackingestimator-1__estimator__alpha':[0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01],
          'stackingestimator-1__estimator__l1_ratio':[0.001,0.003,0.005,0.007,0.01],
          'ridge__alpha':[34.0,35.0,36.0,37.0,38.0,39.0,40.0]}
scoring_fnc = make_scorer(r2_score)
grid= GridSearchCV(stacked_pipeline,params,scoring_fnc)  

grid = grid.fit(finaltrainset, y_train)
model2=grid.best_estimator_

print(model2.get_params()['stackingestimator-1__estimator__alpha'])
#0.01
print(model2.get_params()['stackingestimator-1__estimator__l1_ratio'])
#0.01
print(model2.get_params()['ridge__alpha'])
#40.0

model2.fit(finaltrainset, y_train)
results = model2.predict(finaltestset)



sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56705


#Tune Stacking Estimator_18
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))


params = {'stackingestimator-1__estimator__alpha':[0.01,0.03,0.05,0.08,0.1,0.3,0.5,0.8,1.0,2.0,3.0,4.0,5.0],
          'stackingestimator-1__estimator__l1_ratio':[0.01],
          'ridge__alpha':[40.0,43,45,47,50,53,55,57,60]}
scoring_fnc = make_scorer(r2_score)
grid= GridSearchCV(stacked_pipeline,params,scoring_fnc)  

grid = grid.fit(finaltrainset, y_train)
model2=grid.best_estimator_

print(model2.get_params()['stackingestimator-1__estimator__alpha'])
#1.0
print(model2.get_params()['stackingestimator-1__estimator__l1_ratio'])
#0.01
print(model2.get_params()['ridge__alpha'])
#60

model2.fit(finaltrainset, y_train)
results = model2.predict(finaltestset)



sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56750

#Tune Stacking Estimator_19-best model!!!
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))


params = {'stackingestimator-1__estimator__alpha':[0.9,1.0,1.1,1.2,1.3,1.4,1.5],
          'stackingestimator-1__estimator__l1_ratio':[0.01],
          'ridge__alpha':[60,70,80,90,100,110,120]}
scoring_fnc = make_scorer(r2_score)
grid= GridSearchCV(stacked_pipeline,params,scoring_fnc)  

grid = grid.fit(finaltrainset, y_train)
model2=grid.best_estimator_

print(model2.get_params()['stackingestimator-1__estimator__alpha'])
#1.0
print(model2.get_params()['stackingestimator-1__estimator__l1_ratio'])
#0.01
print(model2.get_params()['ridge__alpha'])
#110

model2.fit(finaltrainset, y_train)
results = model2.predict(finaltestset)



sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56764

#Tune Stacking Estimator_20
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNet(alpha=0.001,l1_ratio=0.1)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    linear_model.Ridge(alpha=31.99))


params = {'stackingestimator-1__estimator__alpha':[1.0],
          'stackingestimator-1__estimator__l1_ratio':[0.01],
          'ridge__alpha':[105,106,107,108,109,110,111,112,113,114,115]}
scoring_fnc = make_scorer(r2_score)
grid= GridSearchCV(stacked_pipeline,params,scoring_fnc)  

grid = grid.fit(finaltrainset, y_train)
model2=grid.best_estimator_

print(model2.get_params()['stackingestimator-1__estimator__alpha'])
#1.0
print(model2.get_params()['stackingestimator-1__estimator__l1_ratio'])
#0.01
print(model2.get_params()['ridge__alpha'])
#109

model2.fit(finaltrainset, y_train)
results = model2.predict(finaltestset)



sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred*0.75 + results*0.25
sub.to_csv('stacked-models.csv', index=False)
#scored  scored  0.56764
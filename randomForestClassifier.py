import data_explore as de
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.externals import joblib



if __name__ == '__main__':
    data = de.prepareForClassifi(de.initTrainData())

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of RFC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # feature scaling
    # X_train, X_test = de.dataMinMaxScale(X_train, X_test)
    X_train, X_test = de.dataStandardScale(X_train, X_test)

    # At first we use default parm to train the model
    # n_estimators=10,criterion="gini",max_depth=None,min_samples_split=2,min_samples_leaf=1,
    # min_weight_fraction_leaf=0.,max_features="auto",max_leaf_nodes=None,min_impurity_decrease=0.,
    # min_impurity_split=None,bootstrap=True,oob_score=False,n_jobs=1,random_state=None,verbose=0,warm_start=False,
    # class_weight=None

    rfc = RandomForestClassifier()
    # rfc.fit(X_train, y_train)
    # predict_rfc = rfc.predict(X_test)
    # print(classification_report(y_test, predict_rfc))

    # default param's result is below
    #              precision    recall  f1-score   support
    #
    #           0       0.90      0.96      0.93       902
    #           1       0.75      0.49      0.59       198
    #
    # avg / total       0.87      0.88      0.87      1100

    # then use GridSearchCV to tuning the param
    param_grid = {'n_estimators': [100, 200, 300, 400],
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [5, 10, 20, None],
                  'min_samples_split': [2, 3, 5, 10],
                  'min_samples_leaf': [1, 2, 3, 5, 10],
                  'min_weight_fraction_leaf' : [0.],
                  'max_features': [1, 3, 10],
                  'max_leaf_nodes': [None],
                  'min_impurity_decrease': [0.],
                  'min_impurity_split': [None],
                  'bootstrap': [True, False]}

    forest_grid = GridSearchCV(estimator=rfc,
                               param_grid=param_grid,
                               scoring="accuracy",
                               cv=3,  # cross-validation
                               n_jobs=4)  # number of core

    # start = time.clock()
    # print('start time is {0}'.format(start))
    # forest_grid.fit(X_train, y_train)  # fit
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = forest_grid.best_estimator_
    # print("Best Model Parameter: ", forest_grid.best_params_)

    # min max scale
    # cost 5802.166018972291s
    # Best Model Parameter: {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 1, 'max_leaf_nodes': None,
    #             'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
    #             'min_weight_fraction_leaf': 0.0, 'n_estimators': 200}

    #              precision    recall  f1-score   support
    #
    #           0       0.91      0.97      0.94       902
    #           1       0.80      0.55      0.65       198
    #
    # avg / total       0.89      0.89      0.89      1100
    rfc = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=None, max_features=1,
                                 max_leaf_nodes= None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=200)
    # rfc.fit(X_train, y_train)
    # predict_rfc = rfc.predict(X_test)
    # print(classification_report(y_test, predict_rfc))




    # standard scale
    # cost 5632.391304687952s
    # Best Model Parameter: {'bootstrap': True, 'criterion': 'gini', 'max_depth': None, 'max_features': 1, 'max_leaf_nodes': None,
    #             'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 3,
    #             'min_weight_fraction_leaf': 0.0, 'n_estimators': 400}
    #
    #             precision    recall  f1-score   support
    #
    #           0       0.90      0.97      0.94       902
    #           1       0.81      0.52      0.63       198
    #
    # avg / total       0.89      0.89      0.88      1100

    rfc = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None, max_features=1,
                                 max_leaf_nodes= None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1,min_samples_split=3,min_weight_fraction_leaf=0.0,n_estimators=400)
    # rfc.fit(X_train, y_train)
    # predict_rfc = rfc.predict(X_test)
    # print(classification_report(y_test, predict_rfc))

    # almost same as the min max scale
    # so let predict the test data
    # use standard scaled data
    testData = de.typeScaling(de.initTestData())

    # X, testData = de.dataStandardScale(X, testData)
    # rfc.fit(X, y)
    # save model to file
    # joblib.dump(rfc, './result/RFCModel_GN.pkl')
    # predict_rfc = rfc.predict(testData)
    # result = pd.Series(predict_rfc)
    # print(result.value_counts())
    # result.to_csv('./result/RFCResult_GN.csv', index=False)

    #0    874
    #1    126


    # then predict the quality rank
    data = de.typeScaling(de.initTrainData())

    y = data['quality']
    X = data.drop('quality', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # feature scaling
    # X_train, X_test = de.dataMinMaxScale(X_train, X_test)
    X_train, X_test = de.dataStandardScale(X_train, X_test)

    rfc = RandomForestClassifier()
    # rfc.fit(X_train, y_train)
    # predict_rfc = rfc.predict(X_test)
    # print(classification_report(y_test, predict_rfc))

    # StandardScale
    #              precision    recall  f1-score   support
    #
    #           3       0.00      0.00      0.00         5
    #           4       0.40      0.13      0.20        31
    #           5       0.66      0.69      0.68       390
    #           6       0.64      0.71      0.67       476
    #           7       0.59      0.52      0.55       159
    #           8       0.88      0.39      0.55        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.64      0.64      0.63      1100

    # min max scaled
    #              precision    recall  f1-score   support
    #
    #           3       0.00      0.00      0.00         5
    #           4       0.20      0.10      0.13        31
    #           5       0.67      0.69      0.68       390
    #           6       0.63      0.70      0.66       476
    #           7       0.59      0.52      0.55       159
    #           8       0.93      0.34      0.50        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.63      0.64      0.63      1100


    param_grid = {'n_estimators': [100, 200, 300, 400],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [5, 10, 20, None],
                      'min_samples_split': [2, 3, 5, 10],
                      'min_samples_leaf': [1, 2, 3, 5, 10],
                      'min_weight_fraction_leaf' : [0.],
                      'max_features': [1, 3, 10],
                      'max_leaf_nodes': [None],
                      'min_impurity_decrease': [0.],
                      'min_impurity_split': [None],
                      'bootstrap': [True, False]}

    # forest_grid = GridSearchCV(estimator=rfc,
    #                            param_grid=param_grid,
    #                            scoring="accuracy",
    #                            cv=3,  # cross-validation
    #                            n_jobs=4)  # number of core
    #
    # start = time.clock()
    # print('start time is {0}'.format(start))
    # forest_grid.fit(X_train, y_train)  # fit
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = forest_grid.best_estimator_
    # print("Best Model Parameter: ", forest_grid.best_params_)

    # min max scaled grid search
    # cost 7690.306514894764s
    # Best Model Parameter:  {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 20, 'max_features': 1,
    # 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 3,
    # 'min_samples_split': 3, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 400}
    #              precision    recall  f1-score   support

    #           3       0.00      0.00      0.00         5
    #           4       0.00      0.00      0.00        31
    #           5       0.72      0.67      0.70       390
    #           6       0.60      0.80      0.69       476
    #           7       0.66      0.38      0.48       159
    #           8       1.00      0.13      0.23        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.64      0.65      0.62      1100

    # standard scaled grid search
    # Best Model Parameter:  {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 20, 'max_features': 1,
    # 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
    # 'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 300}
    #              precision    recall  f1-score   support
    #
    #           3       0.00      0.00      0.00         5
    #           4       0.00      0.00      0.00        31
    #           5       0.74      0.68      0.71       390
    #           6       0.63      0.81      0.71       476
    #           7       0.71      0.48      0.57       159
    #           8       1.00      0.32      0.48        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.67      0.67      0.66      1100

    # the best model of predicting quality rank is standard scaled
    # Best Model Parameter:  {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 20, 'max_features': 1,
    # 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
    # 'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 300}


    rfc = RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=20, max_features=1,
                                 max_leaf_nodes= None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1,min_samples_split=5,min_weight_fraction_leaf=0.0,n_estimators=400,
                                 verbose=True)
    # rfc.fit(X_train, y_train)
    # predict_rfc = rfc.predict(X_test)
    # print(classification_report(y_test, predict_rfc))

    testData = de.typeScaling(de.initTestData())

    X, testData = de.dataStandardScale(X, testData)
    rfc.fit(X, y)
    # save model to file
    joblib.dump(rfc, './result/RFCModel.pkl')
    predict_rfc = rfc.predict(testData)
    result = pd.Series(predict_rfc)
    print(result.value_counts())
    testData = de.initTestData()
    testData.insert(12, 'quality', result)
    testData.to_csv('./result/RFCResult.csv', index=False)

    # 6    567
    # 5    310
    # 7    112
    # 8      8
    # 4      3
import data_explore as de
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import time
from sklearn.externals import joblib

if __name__ == '__main__':
    # do the same thing as RFC
    data = de.prepareForClassifi(de.initTrainData())

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of SGDC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # feature scaling
    # X_train, X_test = de.dataMinMaxScale(X_train, X_test)
    X_train, X_test = de.dataStandardScale(X_train, X_test)

    # start training with default param
    sgdc = SGDClassifier()
    # sgdc.fit(X_train, y_train)
    # predict_sgdc = sgdc.predict(X_test)
    # print(classification_report(y_test, predict_sgdc))
    # default param
    #              precision    recall  f1-score   support
    #
    #           0       0.83      1.00      0.90       902
    #           1       0.67      0.04      0.08       198
    #
    # avg / total       0.80      0.82      0.75      1100

    # then use GridSearchCV to tuning the param, do the same thing as RFC
    # SVM, Logistic Regression, Least-Squares, Boosting
    param_grid = {'loss': ['hinge', 'log', 'squared_loss', 'epsilon_insensitive'],
                  'penalty': ['l2', 'l1', 'elasticnet'], #ridge, lasso, elasticnet
                  'alpha': [ 0.00001, 0.0001, 0.001, 0.01, 1],
                  'max_iter': [10, 100, 200, 300, 400, 500, 600, 700],
                  'learning_rate':['optimal']}

    # sgdc_grid = GridSearchCV(estimator=sgdc,
    #                          param_grid=param_grid,
    #                          scoring="accuracy",
    #                          cv=3,  # cross-validation
    #                          n_jobs=4)  # number of core
    #
    # start = time.clock()
    # sgdc_grid.fit(X_train, y_train)
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = sgdc_grid.best_estimator_
    # print("Best Model Parameter: ", sgdc_grid.best_params_)
    # grid search result
    # Best Model Parameter:  {'alpha': 1e-05, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': 200, 'penalty': 'l1'}
    # sgdc = SGDClassifier(loss='log', alpha=0.00001, max_iter=200, penalty='l1', verbose=True)
    # sgdc.fit(X_train, y_train)
    # predict_sgdc = sgdc.predict(X_test)
    # print(classification_report(y_test, predict_sgdc))
    # as the result 1% better than default param
    #             precision    recall  f1-score   support
    #
    #           0       0.85      0.98      0.91       902
    #           1       0.67      0.19      0.29       198
    #
    # avg / total       0.81      0.84      0.80      1100

    # now use standard scale to train the sgdc model
    # sgdc.fit(X_train, y_train)
    # predict_sgdc = sgdc.predict(X_test)
    # print(classification_report(y_test, predict_sgdc))
    # seems like worse than the minmax scale
    #             precision    recall  f1-score   support
    #
    #           0       0.87      0.84      0.86       902
    #           1       0.38      0.44      0.41       198
    #
    # avg / total       0.78      0.77      0.78      1100
    #
    # then do the grid search
    # sgdc_grid = GridSearchCV(estimator=sgdc,
    #                          param_grid=param_grid,
    #                          scoring="accuracy",
    #                          cv=3,  # cross-validation
    #                          n_jobs=4)  # number of core
    #
    # start = time.clock()
    # sgdc_grid.fit(X_train, y_train)
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = sgdc_grid.best_estimator_
    # print("Best Model Parameter: ", sgdc_grid.best_params_)
    # Best Model Parameter:  {'alpha': 0.0001, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': 300, 'penalty': 'l2'}
    # sgdc = SGDClassifier(loss='log', alpha=0.0001, max_iter=300, penalty='l2', verbose=True)
    # sgdc.fit(X_train, y_train)
    # predict_sgdc = sgdc.predict(X_test)
    # print(classification_report(y_test, predict_sgdc))
    #              precision    recall  f1-score   support
    #
    #           0       0.86      0.95      0.90       902
    #           1       0.56      0.26      0.36       198
    #
    # avg / total       0.80      0.83      0.80      1100

    # so the best of SGDC is
    # minmax scaled data and loss = log penalty = l1
    # now use this model to predict the data in wine_test.csv
    testData = de.typeScaling(de.initTestData())

    X, testData = de.dataMinMaxScale(X, testData)

    sgdc = SGDClassifier(loss='log', alpha=0.00001, max_iter=200, penalty='l1', verbose=True)
    sgdc.fit(X, y)
    # save model to file
    joblib.dump(sgdc, './result/SGDCModel.pkl')
    predict_sgdc = sgdc.predict(testData)
    result = pd.Series(predict_sgdc)
    print(result.value_counts())
    result.to_csv('./result/SGDCResult.csv', index=False)
    # 0    950
    # 1     50




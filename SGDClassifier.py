import data_explore
import randomForestClassifier as RFC
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import time
from sklearn.externals import joblib

if __name__ == '__main__':
    # do the same thing as RFC
    data = RFC.prepareRFC(data_explore.initNormailzedTrainData())
    testData = data_explore.initNormailzedTestData()

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of SGDC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    sgdc = SGDClassifier()
    # sgdc.fit(X_train, y_train)
    # predict_sgdc = sgdc.predict(X_test)
    # print(classification_report(y_test, predict_sgdc))
    # default param
    #              precision    recall  f1-score   support
    #
    #           0       0.87      0.93      0.90       902
    #           1       0.51      0.34      0.41       198
    #
    # avg / total       0.80      0.82      0.81      1100

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
    # cost 77.72111787655928s
    # Best Model Parameter:  {'alpha': 1e-05, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': 600, 'penalty': 'l1'}
    sgdc = SGDClassifier(loss='log', alpha=0.00001, max_iter=600, penalty='l1')
    # sgdc.fit(X_train, y_train)
    # predict_sgdc = sgdc.predict(X_test)
    # print(classification_report(y_test, predict_sgdc))
    # as the result 1% better than default param
    #              precision    recall  f1-score   support
    #
    #           0       0.86      0.94      0.90       902
    #           1       0.55      0.32      0.41       198
    #
    # avg / total       0.81      0.83      0.81      1100

    # now use this model to predict the data in wine_test.csv
    sgdc.fit(X, y)
    # save model to file
    joblib.dump(sgdc, './result/SGDCModel.pkl')

    predict_rfc = sgdc.predict(testData)
    result = pd.Series(predict_rfc)
    print(result.value_counts())
    # 0   956
    # 1   44
    result.to_csv('./result/SGDCResult.csv', index=False)
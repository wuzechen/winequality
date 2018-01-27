import data_explore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.externals import joblib

def prepareRFC(data):
    # Follow to the 80/20 Rule, I think top 20% means good wine and rest 80% means normal wine
    # top 20% of 5497 data is 1095.8 => 1095
    # data = data.sort_values(by='quality', ascending=False)
    # print(data.head(1095))
    # the 1095th data's quality is 0.5000, so quality > 0.5 is 1 means good and quality <= 0.5 means normal
    data['quality'] = pd.cut(data['quality'], bins=[-0.1, 0.5, 1], labels=[0, 1], right=True)
    # print(data['quality'].value_counts())
    # sns.countplot(data['quality'])
    # plt.show()
    return data


if __name__ == '__main__':
    data = prepareRFC(data_explore.initNormailzedTrainData())
    testData = data_explore.initNormailzedTestData()

    y = data['quality']
    X = data.drop('quality', axis=1)
    # y_test = testData['quality']

    # use 20% data for testing the accuracy of RFC
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # At first we use default parm to train the model
    # n_estimators=10,criterion="gini",max_depth=None,min_samples_split=2,min_samples_leaf=1,
    # min_weight_fraction_leaf=0.,max_features="auto",max_leaf_nodes=None,min_impurity_decrease=0.,
    # min_impurity_split=None,bootstrap=True,oob_score=False,n_jobs=1,random_state=None,verbose=0,warm_start=False,
    # class_weight=None

    # rfc = RandomForestClassifier()
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
    # param_grid = {'n_estimators': [100, 200, 300, 400],
    #               'criterion': ['gini', 'entropy'],
    #               'max_depth': [5, 10, 20, None],
    #               'min_samples_split': [2, 3, 5, 10],
    #               'min_samples_leaf': [1, 2, 3, 5, 10],
    #               'min_weight_fraction_leaf' : [0.],
    #               'max_features': [1, 3, 10],
    #               'max_leaf_nodes': [None],
    #               'min_impurity_decrease': [0.],
    #               'min_impurity_split': [None],
    #               'bootstrap': [True, False]}

    # forest_grid = GridSearchCV(estimator=rfc,
    #                            param_grid=param_grid,
    #                            scoring="accuracy",
    #                            cv=3,  # cross-validation
    #                            n_jobs=4)  # number of core

    # start = time.clock()
    # print('start time is {0}'.format(start))
    # forest_grid.fit(X_train, y_train)  # fit
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = forest_grid.best_estimator_
    # print("Best Model Parameter: ", forest_grid.best_params_)

    # cost 5645.481388030462s => 94min
    # Best Model Parameter:  {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 1,
    # 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
    # 'min_samples_split': 5, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 400}
    # sampling without replacement

    rfc = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=None, max_features=1,
                                 max_leaf_nodes= None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1,min_samples_split=5,min_weight_fraction_leaf=0.0,n_estimators=400)
    # rfc.fit(X_train, y_train)
    # predict_rfc = rfc.predict(X_test)
    # print(classification_report(y_test, predict_rfc))

    # a little bit better than the default param
    #              precision    recall  f1-score   support
    #
    #           0       0.91      0.97      0.94       902
    #           1       0.82      0.54      0.65       198
    #
    # avg / total       0.89      0.90      0.89      1100

    # now use this model to predict the data in wine_test.csv
    rfc.fit(X, y)
    # save model to file
    joblib.dump(rfc, './result/RFCModel.pkl')

    predict_rfc = rfc.predict(testData)
    result = pd.Series(predict_rfc)
    print(result.value_counts())
    # 0   983
    # 1   17
    result.to_csv('./result/RFCResult.csv', index=False)
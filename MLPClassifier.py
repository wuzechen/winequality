import data_explore as de
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import time
import pandas as pd
from sklearn.externals import joblib

if __name__ == '__main__':
    # do the same thing as RFC & SGDC
    data = de.prepareForClassifi(de.initTrainData())

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of NN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # feature scaling
    # X_train, X_test = de.dataMinMaxScale(X_train, X_test)
    X_train, X_test = de.dataStandardScale(X_train, X_test)

    nn = MLPClassifier()
    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))

    # the default param result
    #                  precision    recall  f1-score   support
    #
    #               0       0.87      0.95      0.91       902
    #               1       0.59      0.33      0.42       198
    #
    #     avg / total       0.82      0.84      0.82      1100


    # then use GridSearchCV to tuning the param
    # number of nodes per layer = input node +1
    param_grid = {'hidden_layer_sizes':[(100,), (13,13), (13,13,13), (13,13,13,13)],
                  'activation':['identity', 'logistic', 'tanh', 'relu'],
                  'solver':['sgd', 'adam'], #lbfgs is fit data smaller than 1k, so we do not try it
                  'alpha':[0.0001, 0.001],
                  'learning_rate':['constant', 'invscaling', 'adaptive'], #only work for sgd
                  'learning_rate_init':[0.001, 0.0001],
                  'power_t':[0.5],#only work for sgd and invscaling
                  'max_iter':[800, 1000, 1500, 2000]}
    # during the training, there are warnings Maximum iterations (200) reached and the optimization hasn't converged yet.
    # 200 seems too small so add 800 and 1000 in max_iter
    # again 600 is also too small, may be it means 0.00001 learning_rate_init is too small to train it,
    # so del 0.00001 learning rate and add max iter

    # nn_grid = GridSearchCV(estimator=nn,
    #                        param_grid=param_grid,
    #                        scoring="accuracy",
    #                        cv=3,  # cross-validation
    #                        n_jobs=4)  # number of core
    #
    # start = time.clock()
    # nn_grid.fit(X_train, y_train)
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = nn_grid.best_estimator_
    # print("Best Model Parameter: ", nn_grid.best_params_)

    # Best Model Parameter:  {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (13, 13, 13),
    # 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 2000, 'power_t': 0.5, 'solver': 'adam'}


    # nn = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(13, 13, 13),learning_rate='adaptive',
    #                    learning_rate_init=0.001, max_iter=2000, solver='adam', verbose=True)
    #
    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))

    # 13 * 13 * 13 => may be too small, but before try a new size , try standard scale first
    #              precision    recall  f1-score   support
    #
    #           0       0.86      0.94      0.90       902
    #           1       0.56      0.33      0.41       198
    #
    # avg / total       0.81      0.83      0.81      1100

    # nn_grid = GridSearchCV(estimator=nn,
    #                        param_grid=param_grid,
    #                        scoring="accuracy",
    #                        cv=3,  # cross-validation
    #                        n_jobs=4)  # number of core
    #
    # start = time.clock()
    # nn_grid.fit(X_train, y_train)
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = nn_grid.best_estimator_
    # print("Best Model Parameter: ", nn_grid.best_params_)

    #Best Model Parameter:  {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (13, 13, 13, 13),
    # 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 800, 'power_t': 0.5, 'solver': 'adam'}

    #              precision    recall  f1-score   support
    #
    #           0       0.88      0.94      0.91       902
    #           1       0.60      0.39      0.48       198
    #
    # avg / total       0.83      0.84      0.83      1100


    # nn = MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=(1000, 1000, 1000),
    #                    learning_rate='adaptive', learning_rate_init=0.001, max_iter=800, solver='adam', verbose=True)
    #
    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))

    #activation='tanh', alpha=0.001, hidden_layer_sizes=(1000, 1000, 1000, 1000),learning_rate='adaptive',
    # learning_rate_init=0.001, max_iter=800, solver='adam',
    #              precision    recall  f1-score   support
    #
    #           0       0.92      0.92      0.92       902
    #           1       0.64      0.62      0.63       198
    #
    # avg / total       0.87      0.87      0.87      1100

    # activation='relu', alpha=0.001, hidden_layer_sizes=(1000, 1000, 1000),
    # learning_rate='adaptive', learning_rate_init=0.001, max_iter=800, solver='adam'
    #              precision    recall  f1-score   support
    #
    #           0       0.92      0.93      0.93       902
    #           1       0.66      0.63      0.65       198
    #
    # avg / total       0.87      0.88      0.88      1100

    # activation='relu', alpha=0.0001, hidden_layer_sizes=(1000, 1000, 1000),
    # learning_rate='adaptive', learning_rate_init=0.001, max_iter=800, solver='adam'
    #              precision    recall  f1-score   support
    #
    #           0       0.93      0.91      0.92       902
    #           1       0.62      0.68      0.65       198
    #
    # avg / total       0.87      0.87      0.87      1100

    # so the best nn is activation='relu', alpha=0.001, hidden_layer_sizes=(1000, 1000, 1000),
    # learning_rate='adaptive', learning_rate_init=0.001, max_iter=800, solver='adam'
    # and let it predict the test data

    # testData = de.typeScaling(de.initTestData())
    #
    # X, testData = de.dataMinMaxScale(X, testData)
    #
    # nn = MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=(1000, 1000, 1000),
    #                    learning_rate='adaptive', learning_rate_init=0.001, max_iter=800, solver='adam', verbose=True)
    # nn.fit(X, y)
    # joblib.dump(nn, './result/nnModel_GN.pkl')
    # predict_nn = nn.predict(testData)
    # result = pd.Series(predict_nn)
    # print(result.value_counts())
    # result.to_csv('./result/nnResult_GN.csv', index=False)
    # 0    876
    # 1    124


    # now predict the quality rank of wine
    data = de.typeScaling(de.initTrainData())

    y = data['quality']
    X = data.drop('quality', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # feature scaling
    X_train, X_test = de.dataMinMaxScale(X_train, X_test)
    # X_train, X_test = de.dataStandardScale(X_train, X_test)

    nn = MLPClassifier()
    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))

    # standard scaled default param
    #              precision    recall  f1-score   support
    #
    #           3       0.00      0.00      0.00         5
    #           4       0.20      0.06      0.10        31
    #           5       0.62      0.69      0.66       390
    #           6       0.55      0.63      0.59       476
    #           7       0.52      0.35      0.42       159
    #           8       0.33      0.03      0.05        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.55      0.57      0.55      1100

    # min max scaled data
    #              precision    recall  f1-score   support
    #
    #           3       0.00      0.00      0.00         5
    #           4       0.00      0.00      0.00        31
    #           5       0.60      0.61      0.60       390
    #           6       0.52      0.70      0.60       476
    #           7       0.45      0.18      0.26       159
    #           8       0.00      0.00      0.00        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.50      0.54      0.51      1100

    # then use GridSearchCV to tuning the param
    param_grid = {'hidden_layer_sizes':[(1000, 1000), (1000, 1000, 1000), (1000, 1000, 1000)],
                  'activation':['identity', 'logistic', 'tanh', 'relu'],
                  'solver':['sgd', 'adam'], #lbfgs is fit data smaller than 1k, so we do not try it
                  'alpha':[0.0001, 0.001],
                  'learning_rate':['constant', 'invscaling', 'adaptive'], #only work for sgd
                  'learning_rate_init':[0.001, 0.0001],
                  'power_t':[0.5],#only work for sgd and invscaling
                  'max_iter':[800, 1000, 1500, 2000]}

    # nn_grid = GridSearchCV(estimator=nn,
    #                        param_grid=param_grid,
    #                        scoring="accuracy",
    #                        cv=3,  # cross-validation
    #                        n_jobs=4)  # number of core
    #
    # start = time.clock()
    # nn_grid.fit(X_train, y_train)
    #
    # end = time.clock()
    # cost = end - start
    # print('cost {0}s'.format(cost))
    # forest_grid_best = nn_grid.best_estimator_
    # print("Best Model Parameter: ", nn_grid.best_params_)

    # 1000 node per layer cost too much time to grid search, use good / normal model to predict
    nn = MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=(1000, 1000, 1000),
                       learning_rate='adaptive', learning_rate_init=0.001, max_iter=800, solver='adam', verbose=True)

    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))

    # standard scaled data
    #              precision    recall  f1-score   support
    #
    #           3       0.00      0.00      0.00         5
    #           4       0.19      0.10      0.13        31
    #           5       0.69      0.64      0.66       390
    #           6       0.62      0.70      0.66       476
    #           7       0.52      0.52      0.52       159
    #           8       0.48      0.39      0.43        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.61      0.62      0.61      1100

    # min max scaled
    #              precision    recall  f1-score   support
    #
    #           3       0.00      0.00      0.00         5
    #           4       0.19      0.10      0.13        31
    #           5       0.59      0.69      0.63       390
    #           6       0.54      0.53      0.53       476
    #           7       0.41      0.40      0.41       159
    #           8       0.00      0.00      0.00        38
    #           9       0.00      0.00      0.00         1
    #
    # avg / total       0.51      0.53      0.52      1100

    # so standard scaled data is better in classifi

    testData = de.typeScaling(de.initTestData())

    X, testData = de.dataStandardScale(X, testData)

    nn = MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=(1000, 1000, 1000),
                       learning_rate='adaptive', learning_rate_init=0.001, max_iter=800, solver='adam', verbose=True)
    nn.fit(X, y)
    joblib.dump(nn, './result/nnModel.pkl')
    predict_nn = nn.predict(testData)
    result = pd.Series(predict_nn)
    print(result.value_counts())
    testData = de.initTestData()
    testData.insert(12, 'quality', result)
    testData.to_csv('./result/nnResult.csv', index=False)

    # 6    485
    # 5    297
    # 7    160
    # 4     35
    # 8     18
    # 3      4
    # 9      1
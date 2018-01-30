import data_explore
import randomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

if __name__ == '__main__':
    # do the same thing as RFC & SGDC
    data = RFC.prepareRFC(data_explore.initNormailzedTrainData())
    testData = data_explore.initNormailzedTestData()

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of NN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    nn = MLPClassifier()
    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))

    # the default param result
    #              precision    recall  f1-score   support
    #
    #           0       0.86      0.95      0.90       902
    #           1       0.58      0.31      0.41       198
    #
    # avg / total       0.81      0.84      0.81      1100

    # then use GridSearchCV to tuning the param
    # NN is different to SGD, so we split it
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

    # Best Model Parameter: {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'invscaling',
    #             'learning_rate_init': 0.001, 'max_iter': 2000, 'power_t': 0.5, 'solver': 'adam'}

    nn = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(100,),learning_rate='invscaling',
                       learning_rate_init=0.001, max_iter=2000, solver='adam', verbose=True)
    #
    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))

    # the result is not good as expected, may be standard feature scaling is better, so next try standard scaling
    #              precision    recall  f1-score   support
    #
    #           0       0.85      0.97      0.91       902
    #           1       0.64      0.25      0.36       198
    #
    # avg / total       0.82      0.84      0.81      1100

    data = data_explore.initTrainData()
    testData = data_explore.initTestData()
    # quality also need scaled same way 80/20 rule 7 & 8 => 1 , other => 0

    data = data_explore.initTrainData()
    data['type'] = data['type'].apply(lambda x: 1 if x == 'R' else 0)
    data['quality'] = pd.cut(data['quality'], bins=[-0.1, 6, 9], labels=[0, 1], right=True)

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of NN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nn = MLPClassifier()
    # nn.fit(X_train, y_train)
    # predict_nn = nn.predict(X_test)
    # print(classification_report(y_test, predict_nn))
    # default param on standard scaler seems better than min-max scaled data
    #              precision    recall  f1-score   support
    #
    #           0       0.87      0.95      0.91       902
    #           1       0.63      0.36      0.46       198
    #
    # avg / total       0.83      0.85      0.83      1100

    # do the grid search again
    param_grid = {'hidden_layer_sizes':[(100,), (100,100), (100,100,100), (100,100,100,100)],
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

    # Best Model Parameter:  {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100, 100, 100),
    # 'learning_rate': 'invscaling', 'learning_rate_init': 0.0001, 'max_iter': 800, 'power_t': 0.5, 'solver': 'adam'}
    nn = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(1000, 1000, 1000),
                       learning_rate='invscaling', learning_rate_init=0.0001, max_iter=800, solver='adam', verbose=True)
    nn.fit(X_train, y_train)
    predict_nn = nn.predict(X_test)
    print(classification_report(y_test, predict_nn))
    # after 224 iter
    # Iteration 224, loss = 0.14691077
    # Training loss did not improve more than tol=0.000100 for two consecutive epochs. Stopping.
    #              precision    recall  f1-score   support
    #
    #           0       0.91      0.92      0.92       902
    #           1       0.62      0.57      0.59       198
    #
    # avg / total       0.86      0.86      0.86      1100

    #  then i try 1000, 1000, 1000 's NN
    #              precision    recall  f1-score   support
    #
    #           0       0.92      0.93      0.93       902
    #           1       0.66      0.66      0.66       198
    #
    # avg / total       0.88      0.88      0.88      1100



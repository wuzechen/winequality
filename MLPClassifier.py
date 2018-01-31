import data_explore as de
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import time
import pandas as pd

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

    nn_grid = GridSearchCV(estimator=nn,
                           param_grid=param_grid,
                           scoring="accuracy",
                           cv=3,  # cross-validation
                           n_jobs=4)  # number of core

    start = time.clock()
    nn_grid.fit(X_train, y_train)

    end = time.clock()
    cost = end - start
    print('cost {0}s'.format(cost))
    forest_grid_best = nn_grid.best_estimator_
    print("Best Model Parameter: ", nn_grid.best_params_)
import data_explore
import randomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # do the same thing as RFC & SGDC
    data = RFC.prepareRFC(data_explore.initNormailzedTrainData())
    testData = data_explore.initNormailzedTestData()

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of NN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    nn = MLPClassifier()
    nn.fit(X_train, y_train)
    predict_nn = nn.predict(X_test)
    print(classification_report(y_test, predict_nn))

    # the default param result
    #              precision    recall  f1-score   support
    #
    #           0       0.86      0.95      0.90       902
    #           1       0.58      0.31      0.41       198
    #
    # avg / total       0.81      0.84      0.81      1100

    # then use GridSearchCV to tuning the param, do the same thing as RFC & sgdc
    # SVM, Logistic Regression, Least-Squares, Boosting
    #
    param_grid = {'hidden_layer_sizes':[(100,), (13,13,13)],
                  'activation':['identity', 'logistic', 'tanh', 'relu'],
                  'solver':['lbfgs', 'sgd', 'adam'],
                  'alpha':[0.0001, 0.001],
                  'batch_size':['auto'],
                  'learning_rate':['constant', 'invscaling', 'adaptive'],
                  'learning_rate_init':[0.001],
                  'power_t':[0.5],
                  'max_iter':[200, 400, 600],
                  }

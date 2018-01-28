import data_explore
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    data = data_explore.initNormailzedTrainData()
    testData = data_explore.initNormailzedTestData()

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    sdgr = SGDRegressor()
    sdgr.fit(X_train, y_train)
    predict_sdgr = sdgr.predict(X_test)
    print(sdgr.score(X_test, y_test))
    print(r2_score(y_test, predict_sdgr))
    print(mean_squared_error(y_test, predict_sdgr))
    print(mean_absolute_error(y_test, predict_sdgr))
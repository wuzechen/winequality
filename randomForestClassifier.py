import data_explore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

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
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    predict_rfc = rfc.predict(X_test)
    print(classification_report(y_test, predict_rfc))





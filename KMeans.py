import data_explore as de
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # do the same thing
    data = de.typeScaling(de.initTrainData())

    y = data['quality']
    X = data.drop('quality', axis=1)

    # use 20% data for testing the accuracy of kmeans
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

    # feature scaling
    # X_train, X_test = de.dataMinMaxScale(X_train, X_test)
    X_train, X_test = de.dataStandardScale(X_train, X_test)

    # we know there are 2 kind of clusters good and normal
    y_pred = KMeans(n_clusters=7,verbose=True).fit_predict(X_train)
    """绘图"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X_train[:, 2], X_train[:, 3], c=y_pred)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("K-means")
    ax.legend(framealpha=0.1)
    plt.show()
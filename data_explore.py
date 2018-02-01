from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing
import certifi
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def initData(path):
    data = pd.read_csv(path)
    # read the raw data
    print('{0} loaded'.format(path))
    return data

def checkNullValue(data):
    # check the data if has null value -> no null value -> clean data
    print("data has null value? {0}".format(data.isnull().any().any()))

def dataCorrelation(data, isSave):
    # from name, maybe fixed.acidity & volatile.acidity & citric.acid are highly correlated -> false
    # maybe free.sulfur.dioxide & total.sulfur.dioxide are highly correlated -> false
    # check the data's correlation
    correlation = data.corr()
    print(correlation)
    # from the correlation table we find
    # density is positive correlate to fixed.acidity & residual.sugar
    # density is negative correlate to alcohol
    # residual.sugar is positive correlate to free.sulfur.dioxide & total.sulfur.dioxide
    # sulphates is positive correlate to chlorides
    # quality is positive correlate to alcohol
    # volatile.acidity is negative correlate to total.sulfur.dioxide

    # save to file
    if isSave:
        correlation.to_csv('./result/data_correlation.csv')
    print('data_correlation.csv has saved')

def dataMinMaxScaling(data, isSave):
    # change wine type w to 0 and r to 1
    data['type'] = data['type'].apply(lambda x: 1 if x=='R' else 0)
    # print(data.head())

    # before upload data to elasticsearch, do the feature scaling
    # set the rang between 0 and 1
    scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler()
    # min_max scale is better in elasticsearch for visualization
    x_scaled = min_max_scaler.fit_transform(data)
    columns = data.columns
    data = DataFrame(x_scaled, columns=columns)
    # print(data.head())

    if isSave:
        # save scaled data to file
        data.to_csv('./result/train_data_min_max_scaled.csv', index=False)
    print('data normalized')
    return data

def dataStandradScaling(data, isSave):
    # change wine type w to 0 and r to 1
    data['type'] = data['type'].apply(lambda x: 1 if x=='R' else 0)
    # print(data.head())

    # before upload data to elasticsearch, do the feature scaling
    # set the rang between 0 and 1
    scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler()
    # min_max scale is better in elasticsearch for visualization
    x_scaled = scaler.fit_transform(data)
    columns = data.columns
    data = DataFrame(x_scaled, columns=columns)
    # print(data.head())

    if isSave:
        # save scaled data to file
        data.to_csv('./result/train_data_standard_scaled.csv', index=False)
    print('data normalized')
    return data

def uploadElasticearch(data):
    # upload to elasticsearch
    es = Elasticsearch(
        ['YOURELASTICSEARCH'],
        use_ssl=True,
        http_auth=('YOURNAME', 'YOURPASS'),
        ca_certs=certifi.where()
    )

    body = []
    for index, row in data.iterrows():
        doc = {}
        doc['type'] = row['type']
        doc['fixed_acidity'] = row['fixed.acidity']
        doc['volatile_acidity'] = row['volatile.acidity']
        doc['citric_acid'] = row['citric.acid']
        doc['residual_sugar'] = row['residual.sugar']
        doc['chlorides'] = row['chlorides']
        doc['free_sulfur_dioxide'] = row['free.sulfur.dioxide']
        doc['total_sulfur_dioxide'] = row['total.sulfur.dioxide']
        doc['density'] = row['density']
        doc['pH'] = row['pH']
        doc['sulphates'] = row['sulphates']
        doc['alcohol'] = row['alcohol']
        doc['quality'] = row['quality']
        line = {'_op_type': 'index',
                '_index': 'wine_data',
                '_type':'wine',
                '_source':doc}
        body.append(line)

    res = bulk(es, actions=body)
    print('{0} data has uploaded to elasticsearch'.format(res))

def initMinMaxTrainData():
    return dataMinMaxScaling(initTrainData(), False)

def initMinMaxTestData():
    return dataMinMaxScaling(initTestData(), False)

def initTestData():
    return initData('./data/wine_test.csv')

def initTrainData():
    return initData('./data/wine_train.csv')

def prepareForClassifi(data):
    # Follow to the 80/20 Rule, I think top 20% means good wine and rest 80% means normal wine
    # top 20% of 5497 data is 1095.8 => 1095
    # data = data.sort_values(by='quality', ascending=False)
    # print(data.head(1095))
    # the 1095th data's quality is 6, so quality > 6 is 1 means good and quality <= 6 means normal
    data['quality'] = pd.cut(data['quality'], bins=[-0.1, 6, 9], labels=[0, 1], right=True)
    return typeScaling(data)

def typeScaling(data):
    data['type'] = data['type'].apply(lambda x: 1 if x == 'R' else 0)
    return data

def dataMinMaxScale(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def dataStandardScale(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def feature_importances(data):
    data = prepareForClassifi(data)

    y = data['quality']
    X = data.drop('quality', axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    rf = RandomForestClassifier()
    rf.fit(X, y)
    print(rf.feature_importances_)
    # type is not important, and the others are almost same
    # [0.00270186  0.06940971  0.08975956  0.07691974  0.08258686  0.10819747
    #  0.07691445  0.08216788  0.11963666  0.08249469  0.08428739  0.12492373]
    return


if __name__ == '__main__':
    data = initTrainData()
    # dataCorrelation(data, False)
    # data = dataMinMaxScaling(data, True)
    # data = dataStandradScaling(data, True)
    data.info()
    # uploadElasticearch(data)
    # now you can check the basic data status on elasticsearch
    # https://5ad49321f5849cb64b080b8849cb7dfb.us-west-2.aws.found.io:9243
    # username wine passwd lifestyleDE
    # find the wine_data_basic dashboard
    feature_importances(data)
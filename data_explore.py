from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing
import certifi

def initData():
    data = pd.read_csv('./data/wine_train.csv')
    # read the raw data
    print(data.head())
    return data

def checkNullValue(data):
    # check the data if has null value -> no null value -> clean data
    print("data has null value? {0}".format(data.isnull().any().any()))

def dataCorrelation(data):
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
    correlation.to_csv('./result/data_correlation.csv')
    print('data_correlation.csv has saved')

def dataNormalization(data):
    # change wine type w to 0 and r to 1
    data['type'] = data['type'].apply(lambda x: 1 if x=='R' else 0)
    print(data.head())

    # before upload data to elasticsearch, do the feature scaling
    # set the rang between 0 and 1
    scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler()
    # min_max scale is better in elasticsearch for visualization
    x_scaled = min_max_scaler.fit_transform(data)
    columns = data.columns
    data = DataFrame(x_scaled, columns=columns)
    print(data.head())

    # save scaled data to file
    data.to_csv('./result/train_data_scaled.csv', index=False)
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



if __name__ == '__main__':
    data = initData()
    dataCorrelation(data)
    data = dataNormalization(data)
    # uploadElasticearch(data)
    # now you can check the basic data status on elasticsearch
    # https://5ad49321f5849cb64b080b8849cb7dfb.us-west-2.aws.found.io:9243
    #
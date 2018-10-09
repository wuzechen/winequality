# winequality
use wine training data to forecast the test data's quality

use two kind of predict ways
1. 2 - values    quality > 6 => 1  quality <= 6 => 0
2. 7 - values from 3 to 9 
3. ~~minimum-deviation~~ it is a pretty good way to measure result like price prediction, etc.. but I do not think it's good way to measure like rank judging

use 3 kind of models
1. linear model SGD => SVM, Logistic Regression, Least-Squares, Boosting
2. non-linear model => random forest
3. MLP neural network

best model => random forest

param

     bootstrap=True, criterion='entropy', max_depth=20, max_features=1,
     max_leaf_nodes= None, min_impurity_decrease=0.0, min_impurity_split=None,
     min_samples_leaf=1,min_samples_split=5,min_weight_fraction_leaf=0.0,n_estimators=400

precision of model

                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.00      0.00      0.00        31
               5       0.74      0.68      0.71       390
               6       0.63      0.81      0.71       476
               7       0.71      0.48      0.57       159
               8       1.00      0.32      0.48        38
               9       0.00      0.00      0.00         1
    
     avg / total       0.67      0.67      0.66      1100

prezi link https://prezi.com/view/zVguYNlRvEAR8R9pjbsQ

elasticsearch link  https://4c2e6ab8714b46f1b0067cd0f3c8d3d6.ap-northeast-1.aws.found.io:9243

    username wine passwd recruit

   find the wine_train_set_basic_distribution dashboard


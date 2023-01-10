# encoding = utf-8
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from Smote import Smote
from deepSuperLearner import DeepSuperLearner

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

def read_file(path):
    """read_file"""
    data = pd.read_csv(path, sep=',', header=None, encoding='gbk')


    datav = list(data.apply(lambda x: x.iloc[0]))
    data.columns = datav
    df = data.drop([0, 0])

    num, cols = data.shape

    print("The number of sample {} article altogether".format(num))

    column = [
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
        'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
        'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30',
        'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40',
        'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50',
        'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60',
        'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70',
        'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80',
        'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90',
        'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100',
        'f101', 'f102', 'f103', 'f104', 'f105',
        'Sex',
        'Age',
        'Urban_rural',
        'Smoker',
        'SmokingIndex',
        'SmokingExtent',
        'Drinking',
        'DrinkingExtent',
        'Flush',
        'Pickled',
        'HotFood',
        'TooothLoss',
        'ToothLossNumber',
        'GICancer',
        'FamilyHistory'
    ]
    train_df_data = df[column]
    train_df_label = df['GroundTruth_bi']
    # Data consolidation
    train_df = pd.concat([train_df_data, train_df_label], axis=1)
    return train_df

def getTxt():
    train_data = read_file('./dataset/train.csv')

    test_data = read_file('./dataset/test.csv')

    train_data1 = read_file('./dataset/train_somte.csv')

    all_data = pd.concat([train_data, train_data1])
    # all_data = pd.concat([train_data])

    train = all_data.dropna(axis=0, how='any')
    test = test_data.dropna(axis=0, how='any')

    train_labels = [int(i) for i in train['GroundTruth_bi'].values]
    test_labels = [int(i) for i in test['GroundTruth_bi'].values]
    return train_data,train,test,train_labels,test_labels

# Calculate the TN
def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))
# Calculate the FP
def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))
# Calculate the FN
def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))
# Calculate the TP
def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))
# Calculate the precision_score
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0
# Calculate the recall_score
def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)

    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0
# Calculate the Auc img
def Auc(y_test, y_score,title):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    # Calculating the value of AUC
    roc_auc = metrics.auc(fpr, tpr)
    # Map area
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    # Add the marginal line
    plt.plot(fpr, tpr, color='black', lw=1)
    # Add the diagonal
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # Add text information
    plt.text(0.5, 0.3, 'ROC curve (area = %0.5f)' % roc_auc)
    # Add the x axis and y axis labels
    plt.xlabel('l-specificity')
    plt.ylabel('sensitivity')
    plt.title(title)
    # The graphics
    plt.show()

# Training and testing data
def model_train(flags):
    train_data,train,test,train_labels,test_labels = getTxt()
    # set training
    train_result = np.array(train)
    x_train = train_result[:, 0:-1].astype('double')
    y_train = train_result[:, -1].astype('int')

    # set test
    test_result = np.array(test)
    x_test = test_result[:, 0:-1].astype('double')
    y_test = test_result[:, -1].astype('int')


    if flags == 'LightGBM':
        parameters = {
            'learning_rate': np.arange(0.01, 0.15, 0.01),
            'n_estimators': [50, 100, 150, 200],
            'bagging_fraction': [0.6, 0.7, 0.8,0.9,0.95],
            'bagging_frequency': [2, 4],
            'cat_smooth': [0, 10, 20],
            'feature_fraction': np.arange(0.6, 1.0, 0.05),
            'max_depth': [-1, 3, 5, 8],
            'num_leaves': [16, 32, 64],
            'reg_alpha': [0.001, 0.004, 0.008],
            'reg_lambda': [2, 4, 6],
        }
        grid = LGBMClassifier()
    elif flags == 'AdaBoost':
        parameters = {
            'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'n_estimators': [50, 100, 150, 200],
            'algorithm': ['SAMME', 'SAMME.R']
        }
        grid = AdaBoostClassifier()
    elif flags == 'XGBoost':
        parameters = {
            'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
            'gamma': np.arange(0.01, 0.1, 0.01),
            'max_depth': [1, 3, 5, 8, 10],
            'min_child_weight': np.arange(1, 6, 1),
            'max_delta_step': [0, 10 , 20],
            'subsample': np.arange(0.7, 1.0, 0.1),
            'colsample_bytree': np.arange(0.7, 1.0, 0.1),
            'colsample_bylevel': np.arange(0.7, 1.0, 0.1),
            'reg_lambda': [5,6,8,10],
            'reg_alpha': [0.001, 0.004, 0.008],
            'n_estimators': [50, 100, 150, 200]
        }
        grid = XGBClassifier()
    elif flags == 'SVM':
        parameters = {
            'C': [0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'sigmoid', 'poly']
        }
        stdsc = StandardScaler()
        x_train = stdsc.fit_transform(x_train)
        x_test = stdsc.fit_transform(x_test)
        grid = svm.SVC(probability=True)
    elif flags == 'RandomForest':
        parameters = {
            'max_depth': [1,3,5,8,10],
            'n_estimators': [50, 100, 150, 200],
            'min_samples_split': [1,5],
            'max_features': range(5, 76, 10)
        }
        grid = RandomForestClassifier()
   
    grid = GridSearchCV(grid, param_grid=parameters, cv=3, scoring='roc_auc', verbose=100, n_jobs=1)

    # Training model in the training set
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    print('------------------------------------- ', flags, ' Start -------------------------------------')
    print('The best value of parameters:{0}'.format(grid.best_params_))
    print('The best score model:{0}'.format(grid.best_score_))
    # Setting up the grid parameters
    grid = grid.best_estimator_
    grid.fit(x_train, y_train)
    # Save the model
    joblib.dump(grid, './model3/' + flags + '.pkl')

    test_predict = grid.predict(x_test)
    confusion_matrix_result = metrics.confusion_matrix(y_test, test_predict)
    print('The confusion matrix result:\n', confusion_matrix_result)

    y_score = grid.predict_proba(x_test)[:, 1]
    print('AUC', metrics.roc_auc_score(y_test, y_score))
    print('APR', metrics.average_precision_score(y_test, y_score))

    print(classification_report(y_test, test_predict))
    Auc(y_test, y_score, flags)

    print('------------------------------------- ', flags, ' End -------------------------------------')

def superLearner():
    LGB_learner = LGBMClassifier(
        cat_smooth=10,
        feature_fraction=0.95,
        learning_rate=0.01,
        max_depth=3,
        num_leaves=16,
        reg_alpha=0.004,
        reg_lambda=6,
        n_estimators=150,
        bagging_fraction=0.6,
        bagging_frequency=4
    )
    Ada_learner = AdaBoostClassifier(n_estimators=150,learning_rate=0.6, algorithm='SAMME.R')
    XGB_learner = XGBClassifier(learning_rate=0.02,
            gamma=0.01,
            max_depth=3,
            min_child_weight=2,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_lambda=0.004,
            reg_alpha=6,
            n_estimators=150)
    RFC_learner = RandomForestClassifier(n_estimators=50, max_depth=5, max_features=35, min_samples_split=2)

    Base_learners = {'LGBMClassifier':LGB_learner, 'AdaBoostClassifier':Ada_learner, 'XGBClassifier':XGB_learner,
                     'RandomForestClassifier':RFC_learner}
    np.random.seed(100)
    train_data, train, test, train_labels, test_labels = getTxt()
    # The training set data numerical value
    train_result = np.array(train)
    x_train = train_result[:, 0:-1].astype('double')
    y_train = train_result[:, -1].astype('int')

    # Test data set numerical value
    test_result = np.array(test)
    x_test = test_result[:, 0:-1].astype('double')
    y_test = test_result[:, -1].astype('int')

    DSL_learner = DeepSuperLearner(Base_learners)
    DSL_learner.fit(x_train, y_train)
    DSL_learner.get_precision_recall(x_test, y_test, show_graphs=True)

    print('------------------------------------- DSL_learner Start -------------------------------------')
    test_predict = DSL_learner.predict(x_test)
    test_predict = numpy.argmax(test_predict, axis=1)
    confusion_matrix_result = metrics.confusion_matrix(y_test, test_predict)
    print('The confusion matrix result:\n', confusion_matrix_result)

    y_score = DSL_learner.predict(x_test)[:, 1]
    print('AUC', metrics.roc_auc_score(y_test, y_score))
    print('APR', metrics.average_precision_score(y_test, y_score))

    print(classification_report(y_test, test_predict))
    Auc(y_test, y_score, 'superLearner')
    print('------------------------------------- DSL_learner End -------------------------------------')


def Smote(n):
    data = pd.read_csv('./dataset/train.csv', sep=',', header=None, encoding='gbk')
    datav = list(data.apply(lambda x: x.iloc[0]))
    data.columns = datav
    df = data.drop([0, 0])
    column = [
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
        'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
        'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30',
        'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40',
        'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50',
        'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60',
        'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70',
        'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80',
        'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90',
        'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100',
        'f101', 'f102', 'f103', 'f104', 'f105']
    train_df_data = df[column]
    train_df_label = df['GroundTruth_bi']
    train_df = pd.concat([train_df_data, train_df_label], axis=1)

    train = train_df.dropna(axis=0, how='any')
    train_result = np.array(train)
    x_train = train_result[:, 0:-1].astype('double')
    y_train = train_result[:, -1].astype('int')
    # Smote for data balancing
    smote = Smote(N=n)
    synthetic_points = smote.fit(x_train)
    numpy.savetxt('train_x.csv', synthetic_points, delimiter=',')

if __name__ == '__main__':
    # Smote(3000)
    # run LightGBM
    model_train('LightGBM')
    # # run AdaBoost
    # model_train('AdaBoost')
    # # run XGBoost
    # model_train('XGBoost')
    # # run SVM
    # model_train('SVM')
    # # run RandomForest
    # model_train('RandomForest')
    # # run superLearner
    # superLearner()



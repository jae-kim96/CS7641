import pandas as pd
import numpy as np
import time
import warnings
from scipy.sparse import data

from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

## Script for Assignment 1 for CS 7641 
## Implementing different algortihms and looking at the performance

## Method for Plotting Validation Curves
def plot_validation_curve(clf, clf_name, X, y, param, param_range, dataset_name):
    
    train_scores, test_scores = validation_curve(clf, 
        X, y, param_name = param, param_range = param_range, scoring = 'accuracy'
    )

    train_mean = np.mean(train_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)

    val_data = {
        'param': param_range,
        'training_score': train_mean,
        'testing_score': test_mean
    }

    val_df = pd.DataFrame(val_data).set_index('param')
    val_df.plot.line()
    plt.title(f'Validation Curve: {dataset_name} - {param}')
    plt.ylabel('Accuracy')
    plt.xlabel(param)
    filename = clf_name + '/' + dataset_name + '-' + param + ' Validation Curve.png'
    plt.savefig(filename)

## Dataset Paths
water_path = 'data/water_potability.csv' ## water potability data
wine_path = 'data/winequality-white.csv' ## wine quality data

## Function to Clean the Wine Quality data
def clean_wine_data():
    wine_df = pd.read_csv(wine_path, sep = ';')

    wine_df['binary_quality'] = np.where(wine_df['quality'] >= 7, 1, 0)
    # print(wine_df['binary_quality'].value_counts())
    X = wine_df.loc[:, 'fixed acidity': 'alcohol']
    y = wine_df.loc[:, 'binary_quality']
    
    print(y.value_counts())
    return X, y

## Function to Clean the Water Potability Data
def clean_water_data():
    water_df = pd.read_csv(water_path).dropna().reset_index(drop = True)
    
    X = water_df.loc[:, 'ph': 'Turbidity']
    y = water_df.loc[:, 'Potability']
    print(y.value_counts())
    return X, y

## Decision Tree Performance for Wine Dataset
def decision_tree(X, y, dataset_name):
    '''
    Varying the training size to create a learning curve on the dataset passed.
    Training Accuracy and Testing Accuracy vs Training Size
    '''
    # training_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    training_acc = list()
    testing_acc = list()

    params = {
        'max_depth': np.arange(2, 20, 1).tolist(),
        'min_samples_leaf': np.arange(1, 50, 5).tolist()
    }

    cv = GridSearchCV(DecisionTreeClassifier(), params)
    cv.fit(X, y)

    best_param = cv.best_params_
    max_depth = best_param['max_depth']
    min_samples_leaf = best_param['min_samples_leaf']

    print(f'Best {dataset_name} Params: max_depth = {max_depth}, min_samples_leaf = {min_samples_leaf}')
    clf = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, random_state = 42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy of DT = {accuracy_score(y_test, y_pred)}')

    ## Looking at the Training Size
    training_sizes = np.arange(0.1, 0.95, 0.01).tolist()
    for s in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = s, random_state = 42)

        clf.fit(X_train, y_train)

        ## Training Accuracy
        y_in_pred = clf.predict(X_train)
        training_acc.append(accuracy_score(y_train, y_in_pred))

        ## Testing Accuracy
        y_pred = clf.predict(X_test)
        testing_acc.append(accuracy_score(y_test, y_pred))


    result = {'training_size': training_sizes, 'testing_accuracy': training_acc, 'training_accuracy': testing_acc}
    result_df = pd.DataFrame(result).set_index('training_size')

    result_df.plot.line()
    filename = 'decision_tree/' + dataset_name + ' - Decision Tree Learning Curve.png'
    plt.savefig(filename)

    ## Looking at Effect of Pruning via Validation Curves
    for p in params.keys():
        plot_validation_curve(DecisionTreeClassifier(), 'decision_tree', X, y, p, params[p],dataset_name)
    
## Neural Network Implementation
@ignore_warnings(category = ConvergenceWarning)
def mlp_classifier(X, y, dataset_name):
    '''
    Multi Layer Perceptron Neural Network as a classifier
    '''

    ## Grid Searching for the best MLP Parameters
    params = {
        'learning_rate_init': [1, 0.1, 0.001, 0.0001],
        'hidden_layer_sizes': [(10,), (50,), (100,), (150,), (200,), (250,)],
        'max_iter': np.arange(10, 250, 10).tolist()
    }

    ## Validation Curves after changing some of the params
    for p in params.keys():
        plot_validation_curve(MLPClassifier(), 'mlp', X, y, p, params[p],dataset_name)
    
    ## Grid Search for best params
    cv = GridSearchCV(MLPClassifier(), params)
    cv.fit(X, y)

    best_param = cv.best_params_
    hidden_layers = best_param['hidden_layer_sizes']
    learning = best_param['learning_rate_init']
    max_iter = best_param['max_iter']

    print(f'For dataset: {dataset_name}, hidden_layers: {hidden_layers}, learning: {learning}, max_iter: {max_iter}')

    ## Actual MLP Classifier after Grid Search
    clf = MLPClassifier(hidden_layer_sizes = hidden_layers, learning_rate_init = learning, max_iter = max_iter)
    ## Accuracy
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, random_state = 42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy of MLP = {accuracy_score(y_test, y_pred)}')

## Boosting Decision Tree Implementation
def boosted_decision_tree(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, random_state = 42)
    dt = DecisionTreeClassifier(max_depth = 4, random_state = 42)

    params = {
        'n_estimators': [10, 25, 50, 100, 200, 500],
        'learning_rate': [0.25, 0.5, 1., 2., 4., 8.]
    }
    cv = GridSearchCV(AdaBoostClassifier(dt), params)
    cv.fit(X, y)

    best_params = cv.best_params_
    n_estimators = best_params['n_estimators']
    learning_rate = best_params['learning_rate']

    print(f'For {dataset_name}, n_estimators = {n_estimators}, learning_rate = {learning_rate}')

    clf = AdaBoostClassifier(dt, n_estimators = n_estimators, learning_rate = learning_rate)
    clf.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, random_state = 42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy of DT = {accuracy_score(y_test, y_pred)}')

    ## Validation Curves
    for p in params.keys():
        plot_validation_curve(AdaBoostClassifier(dt), 'boosted_dt', X, y, p, params[p],dataset_name)

    ## Learning Curves
    training_acc = list()
    testing_acc = list()
    training_sizes = np.arange(0.01, 0.99, 0.01).tolist()
    for s in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = s, random_state = 42)

        clf.fit(X_train, y_train)

        ## Training Accuracy
        y_in_pred = clf.predict(X_train)
        training_acc.append(accuracy_score(y_train, y_in_pred))

        ## Testing Accuracy
        y_pred = clf.predict(X_test)
        testing_acc.append(accuracy_score(y_test, y_pred))


    result = {'training_size': training_sizes, 'testing_accuracy': testing_acc, 'training_accuracy': training_acc}
    result_df = pd.DataFrame(result).set_index('training_size')
    result_df.head()

    result_df.plot.line()
    filename = 'boosted_dt/' + dataset_name + ' - Decision Tree Results.png'
    plt.savefig(filename)

## SVM Impementation
@ignore_warnings(category = ConvergenceWarning)
def svm_classifier(X, y, dataset_name, kernel_type):
    '''
    Support Vector Machine implementation for the datasets
    '''
    ## Standardize data for svm
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    valid_params = {
        'C': np.arange(0.1, 1.0, 0.05).tolist(),
        'max_iter': np.arange(10, 250, 10).tolist()
    }

    ## Generate Validation Curves
    for p in valid_params.keys():
        plot_validation_curve(SVC(kernel = kernel_type), 'svc' + '-' + kernel_type, X_scaled, y, p, valid_params[p], dataset_name)

    ## Grid Search
    params = {
        'C': np.arange(0.1, 1.0, 0.05).tolist(),
        'max_iter': np.arange(10, 250, 10).tolist(),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    cv = GridSearchCV(SVC(), params)
    cv.fit(X_scaled, y)

    C_value = cv.best_params_['C']
    kernel = cv.best_params_['kernel']
    max_iter = cv.best_params_['max_iter']

    print(f'Best Parameters: C = {C_value}, kernel: {kernel}, max_iter: {max_iter}')

    svc = SVC(C = C_value, kernel = kernel, max_iter = max_iter)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size = 0.3, random_state = 42)
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)

    print(f'Accuracy Score = {accuracy_score(y_test, y_pred)}')

## KNN Classifier Analysis
def knn_classifier(X, y, dataset_name):
    '''
    A method for analyzing the performance of the KNN Classifier
    '''
    ## Standardize data for svm
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    ## params
    params = {
        'n_neighbors': np.arange(1, 15, 1).tolist()
    }
    for p in params.keys():
        plot_validation_curve(KNeighborsClassifier(), 'knn', X_scaled, y, p, params[p], dataset_name)
    
    cv = GridSearchCV(KNeighborsClassifier(), params)
    cv.fit(X_scaled, y)

    n = cv.best_params_['n_neighbors']
    
    print(f'Best Parameters: n_neighbors = {n}')

    knn = KNeighborsClassifier(n_neighbors = n)

    ## Learning Curve
    training_sizes = np.arange(0.1, 0.95, 0.01).tolist()
    training_acc = list()
    testing_acc = list()
    for s in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size = s, random_state = 42)

        knn.fit(X_train, y_train)

        ## Training Accuracy
        y_in_pred = knn.predict(X_train)
        training_acc.append(accuracy_score(y_train, y_in_pred))

        ## Testing Accuracy
        y_pred = knn.predict(X_test)
        testing_acc.append(accuracy_score(y_test, y_pred))


    result = {'training_size': training_sizes, 'testing_accuracy': training_acc, 'training_accuracy': testing_acc}
    result_df = pd.DataFrame(result).set_index('training_size')

    result_df.plot.line()
    filename = 'knn/' + dataset_name + ' - KNN Learning Curve.png'
    plt.savefig(filename)

    # Best KNN Accuracy
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size = 0.3, random_state = 42)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print(f'Accuracy Score = {accuracy_score(y_test, y_pred)}')

## Measuring Wall Clock Time and Accuracy
@ignore_warnings(category = ConvergenceWarning)
@ignore_warnings(category = DeprecationWarning)
def modeling_efficiency(X, y, dataset_name):
    ## Parameters are the best params taken from running GridSearchCV above.
    if dataset_name == 'Water Potability':
        models = {
            'decision_tree': DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 36),
            'mlp': MLPClassifier(hidden_layer_sizes = (10,), learning_rate_init = 0.1, max_iter = 190),
            'boosted_tree': AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4), learning_rate = 0.25, n_estimators = 10),
            'svm': SVC(kernel = 'rbf', C = 0.70),
            'knn': KNeighborsClassifier(n_neighbors = 14)
        }
    else:
        models = {
            'decision_tree': DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 11),
            'mlp': MLPClassifier(hidden_layer_sizes = (250,), learning_rate_init = 0.001, max_iter = 240),
            'boosted_tree': AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4), learning_rate = 0.5, n_estimators = 500),
            'svm': SVC(kernel = 'rbf', C = 0.55),
            'knn': KNeighborsClassifier(n_neighbors = 14)
        }

    ## Calculating the Accuracy of each Learner with 0.33 Test Size
    accuracy = dict()
    for m in models.keys():
        if m in ['svm', 'knn']: # Scaling data for knn and svm
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
        clf = models[m]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy[m] = [accuracy_score(y_test, y_pred)]
    
    accuracy_df = pd.DataFrame(accuracy).transpose()
    accuracy_df.plot.bar(rot = 0, ylim = (0.55, 0.9), legend = False)
    plt.title('Accuracy for ' + dataset_name)
    plt.savefig('Accuracy for ' + dataset_name + '.png')
        
    ## Calculate Wall Time for different Training Sample Sizes
    training_sizes = np.arange(0.1, 0.95, 0.05).tolist()
    wall_time = {
        'training_sizes': training_sizes
    }
    for m in models.keys():
        print('Starting to work on Learner = ' + m)
        clf = models[m]
        if m in ['svm', 'knn']: # Scaling data for knn and svm
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
        
        ## Training on different sizes
        training_times = list()
        for s in training_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = s, random_state = 42)
            
            t0 = time.clock()
            clf.fit(X_train, y_train)
            tf = time.time() - t0

            training_times.append(tf)

        # result = {'training_size': training_sizes, 'wall_time': training_times}
        wall_time[m] = training_times
    
    wall_time_df = pd.DataFrame(wall_time).set_index('training_sizes')
    # wall_time_df = (wall_time_df - wall_time_df.mean()) / wall_time_df.std()
    # wall_time_df = (wall_time_df - wall_time_df.min()) / (wall_time_df.max() - wall_time_df.min())
    wall_time_df.plot.line()
    plt.yscale('symlog')
    plt.title('Wall Time - ' + dataset_name)
    plt.ylabel('Time in Seconds (Log)')
    plt.savefig('Wall Time - ' + dataset_name + '.png')


## Running the scripts
if __name__ == "__main__":
    ## Getting the X and y for the classification Dataset
    wine_X, wine_y = clean_wine_data()
    water_X, water_y = clean_water_data()

    ## Decision Tree Learning Curve
    print('============WORKING on DECISION TREE ANALYSIS============')
    decision_tree(wine_X, wine_y, 'Wine Quality')
    decision_tree(water_X, water_y, 'Water Potability')
    print('============FINISHED DECISION TREE ANALYSIS============')

    ## MLP Classifier Learning Curve
    print('============WORKING on MLP Classifier ANALYSIS============')
    mlp_classifier(wine_X, wine_y, 'Wine Quality')
    mlp_classifier(water_X, water_y, 'Water Potability')
    print('============FINISHED MLP Classifier ANALYSIS============')

    ## Boosted Decision Tree
    print('============WORKING on Boosted Decision Tree Classifier ANALYSIS============')
    boosted_decision_tree(wine_X, wine_y, 'Wine Quality')
    boosted_decision_tree(water_X, water_y, 'Water Potability')
    print('============FINISHED Boosted Decision Tree Classifier ANALYSIS============')

    ## SVM Classifier
    print('============WORKING on Support Vector Machine Classifier ANALYSIS============')
    svm_classifier(wine_X, wine_y, 'Wine Quality', 'rbf')
    svm_classifier(water_X, water_y, 'Water Potability', 'rbf')
    svm_classifier(wine_X, wine_y, 'Wine Quality', 'sigmoid')
    svm_classifier(water_X, water_y, 'Water Potability', 'sigmoid')
    print('============FINISHED Support Vector Machine Classifier ANALYSIS============')

    ## KNN Classifier
    print('============WORKING on K-Nearest Neighbors Classifier ANALYSIS============')
    knn_classifier(wine_X, wine_y, 'Wine Quality')
    knn_classifier(water_X, water_y, 'Water Potability')
    print('============FINISHED K-Nearest Neighbors Classifier ANALYSIS============')

    ## Calculate Wall Time
    modeling_efficiency(wine_X, wine_y, 'Wine Quality')
    modeling_efficiency(water_X, water_y, 'Water Potability')
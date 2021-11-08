import pandas as pd
import numpy as np
import time
import math

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score, log_loss
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt


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
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # print(X.shape)
    print(y.value_counts())
    return X_scaled, y

## MLPClassifier on reduced dataset
def nn_classifier(X, y):
    dim_reductions = {
        'pca': PCA(n_components = 4),
        'ica': FastICA(n_components = 11, max_iter = 800),
        'rp': GaussianRandomProjection(n_components = 8),
        'kpca': KernelPCA(n_components = 6, kernel = 'poly')
    }

    num_iters = np.arange(1, 201, 1).tolist()

    ## Wall Times
    wall_times = {
        'num_iters': num_iters,
        'regular': list()
    }

    ## Wall time for Regular NN
    for iter in num_iters:
        mlp = MLPClassifier(
            hidden_layer_sizes = (32, 64,  8, 16,), 
            learning_rate_init = 0.01,
            activation = 'tanh', 
            max_iter = iter,
            solver = 'sgd',
            learning_rate = 'adaptive'
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        ## Training Time
        t0 = time.time()
        mlp.fit(X_train, y_train)
        train_time = time.time() - t0
        wall_times['regular'].append(train_time)

    for red in dim_reductions.keys():
        print(f'--> RUNNING {red}')
        ## Reducing the Data and Splitting into Testing Sets
        
        curr = dim_reductions[red]
        X_red = curr.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size = 0.3, random_state = 42)

        times = list()
        train_acc = list()
        test_acc = list()
        train_loss = list()
        test_loss = list()

        for iter in num_iters:
            mlp = MLPClassifier(
                hidden_layer_sizes = (32, 64,  8, 16,), 
                learning_rate_init = 0.01,
                activation = 'tanh', 
                max_iter = iter,
                solver = 'sgd',
                learning_rate = 'adaptive'
            )

            ## Training Time
            t0 = time.time()
            mlp.fit(X_train, y_train)
            train_time = time.time() - t0
            # times.append(math.log(train_time))
            times.append(train_time)
            
            ## Training Accuracy and Loss
            y_pred_in = mlp.predict(X_train)
            train_acc.append(accuracy_score(y_train, y_pred_in))
            train_loss.append(log_loss(y_train, y_pred_in))

            ## Testing Accuracy and Loss
            y_pred = mlp.predict(X_test)
            test_acc.append(accuracy_score(y_test, y_pred))
            test_loss.append(log_loss(y_test, y_pred))

        ## Training Times
        wall_times[red] = times

        ## Plotting Accuracy for Dim Reduction
        acc_df = pd.DataFrame({'num_iters': num_iters, 'training_accuracy': train_acc, 'testing_accuracy': test_acc}).set_index('num_iters')
        acc_df.plot()
        plt.title(f'{red}-MLP Accuracy Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig(f'part4/{red}- Accuracy Curve.png')
        plt.close()

        ## Plotting Loss for Dim Reduction
        acc_df = pd.DataFrame({'num_iters': num_iters, 'training_loss': train_loss, 'testing_loss': test_loss}).set_index('num_iters')
        acc_df.plot()
        plt.title(f'{red}-MLP Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Log Loss')
        plt.savefig(f'part4/{red}- Loss Curve.png')
        plt.close()

    ## Plotting Wall Times over Iterations
    wall_times_df = pd.DataFrame(wall_times).set_index('num_iters')
    wall_times_df.plot()
    plt.title('Training Time Curves for Dimensionality Reductions')
    plt.xlabel('Iterations')
    plt.ylabel('Train Time')
    plt.savefig('part4/Wall Times.png')
    plt.close()

    ## Last Wall Time for 200 iterations
    last_df = wall_times_df.iloc[-1:]
    last_df.plot.bar()
    plt.title('Training Times for Dimensionality Reductions')
    plt.ylabel('Train Time in Seconds')
    plt.savefig('part4/Wall Times Bar Plot.png')
    plt.close()



if __name__ == "__main__":
    wine_X, wine_y = clean_wine_data()

    print('===========================Dimensionality Reduction and NN Wine Dataset===========================')
    nn_classifier(wine_X, wine_y)
import pandas as pd
import numpy as np
import time
import math

from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score, log_loss

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

## Running the NN on the regular dataset
def nn(X, y):
    '''
    Neural Network without any Dim Reduction or Clustering.
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    num_iters = np.arange(1, 201, 1).tolist()
    train_acc = list()
    test_acc = list()
    train_loss = list()
    test_loss = list()

    for iter in num_iters:
        mlp = MLPClassifier(
            # hidden_layer_sizes = (32, 32, 32, 16, 8, 4,),
            # hidden_layer_sizes = (64, 32, 32, 32, 16, 4,), ok
            hidden_layer_sizes = (32, 64,  8, 16,), ##good
            # hidden_layer_sizes = (32, 64,  8, 16, 2, 4,), ## good
            learning_rate_init = 0.01,
            activation = 'tanh', ## good
            max_iter = iter,
            solver = 'sgd',
            learning_rate = 'adaptive'
        )
        mlp.fit(X_train, y_train)
        # ## Training Time
        # t0 = time.time()
        # mlp.fit(X_train, y_train)
        # train_time = time.time() - t0
        # times.append(train_time)
        
        ## Training Accuracy and Loss
        y_pred_in = mlp.predict(X_train)
        train_acc.append(accuracy_score(y_train, y_pred_in))
        train_loss.append(log_loss(y_train, y_pred_in))

        ## Testing Accuracy and Loss
        y_pred = mlp.predict(X_test)
        test_acc.append(accuracy_score(y_test, y_pred))
        test_loss.append(log_loss(y_test, y_pred))

    ## Plotting Accuracy for Dim Reduction
    acc_df = pd.DataFrame({'num_iters': num_iters, 'training_accuracy': train_acc, 'testing_accuracy': test_acc}).set_index('num_iters')
    acc_df.plot()
    plt.title(f'MLP Accuracy Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig(f'part5/MLP - Accuracy Curve.png')
    plt.close()

    ## Plotting Loss for Dim Reduction
    loss_df = pd.DataFrame({'num_iters': num_iters, 'training_loss': train_loss, 'testing_loss': test_loss}).set_index('num_iters')
    loss_df.plot()
    plt.title(f'MLP Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.savefig(f'part5/MLP - Loss Curve.png')
    plt.close()


## CLustering and NN Experiment
def nn_classifier(X, y):
    '''
    CLustering and Neural Network Training
    '''
    ## Wall Times
    num_iters = np.arange(1, 201,1).tolist()
    wall_times = {'num_iters': num_iters}

    ## Normal NN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    num_iters = np.arange(1, 201, 1).tolist()
    train_acc = list()
    test_acc = list()
    train_loss = list()
    test_loss = list()
    nn_times = list()

    for iter in num_iters:
        mlp = MLPClassifier(
            # hidden_layer_sizes = (32, 32, 32, 16, 8, 4,),
            # hidden_layer_sizes = (64, 32, 32, 32, 16, 4,), ok
            hidden_layer_sizes = (32, 64,  8, 16,), ##good
            # hidden_layer_sizes = (32, 64,  8, 16, 2, 4,), ## good
            learning_rate_init = 0.01,
            activation = 'tanh', ## good
            max_iter = iter,
            solver = 'sgd',
            learning_rate = 'adaptive'
        )
        mlp.fit(X_train, y_train)
        ## Training Time
        t0 = time.time()
        mlp.fit(X_train, y_train)
        train_time = time.time() - t0
        nn_times.append(train_time)
        
        ## Training Accuracy and Loss
        y_pred_in = mlp.predict(X_train)
        train_acc.append(accuracy_score(y_train, y_pred_in))
        train_loss.append(log_loss(y_train, y_pred_in))

        ## Testing Accuracy and Loss
        y_pred = mlp.predict(X_test)
        test_acc.append(accuracy_score(y_test, y_pred))
        test_loss.append(log_loss(y_test, y_pred))

    wall_times['normal'] = nn_times

    ## Plotting Accuracy for Dim Reduction
    acc_df = pd.DataFrame({'num_iters': num_iters, 'training_accuracy': train_acc, 'testing_accuracy': test_acc}).set_index('num_iters')
    acc_df.plot()
    plt.title(f'MLP Accuracy Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig(f'part5/MLP - Accuracy Curve.png')
    plt.close()

    ## Plotting Loss for Dim Reduction
    loss_df = pd.DataFrame({'num_iters': num_iters, 'training_loss': train_loss, 'testing_loss': test_loss}).set_index('num_iters')
    loss_df.plot()
    plt.title(f'MLP Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.savefig(f'part5/MLP - Loss Curve.png')
    plt.close()

    ## CLustering Algorithms on NN
    clustering = {
        'kmeans': KMeans(n_clusters = 2),
        'gmm': GaussianMixture(n_components = 3)
    }

    for cluster in clustering.keys():
        print(f'--> RUNNING {cluster}')
        ## Using labels as features
        curr = clustering[cluster]
 
        if cluster == 'kmeans':
            curr.fit(X)
            ## Using labels
            # labels = curr.predict(X)
            # X_new = pd.DataFrame(X)
            # X_new['labels'] = labels


            ## Try fit Transform or adding cluster labels
            X_cluster = curr.fit_transform(X)
            X_new = np.append(X, X_cluster, axis = 1)
           
            # X_new = np.append(X, labels, axis = 1)
        else:
            curr.fit(X)
            X_prob = curr.predict_proba(X)
            X_new = np.append(X, X_prob, axis = 1)

        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state = 42)

        times = list()
        train_acc = list()
        test_acc = list()
        train_loss = list()
        test_loss = list()

        for iter in num_iters:
            mlp = MLPClassifier(
                hidden_layer_sizes = (32, 64, 8, 16,),
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
        wall_times[cluster] = times

        ## Plotting Accuracy for Dim Reduction
        acc_df = pd.DataFrame({'num_iters': num_iters, 'training_accuracy': train_acc, 'testing_accuracy': test_acc}).set_index('num_iters')
        acc_df.plot()
        plt.title(f'{cluster}-MLP Accuracy Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig(f'part5/{cluster}- Accuracy Curve.png')
        plt.close()

        ## Plotting Loss for Dim Reduction
        acc_df = pd.DataFrame({'num_iters': num_iters, 'training_loss': train_loss, 'testing_loss': test_loss}).set_index('num_iters')
        acc_df.plot()
        plt.title(f'{cluster}-MLP Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Log Loss')
        plt.savefig(f'part5/{cluster}- Loss Curve.png')
        plt.close()

    ## Plotting Wall Times over Iterations
    wall_times_df = pd.DataFrame(wall_times).set_index('num_iters')
    wall_times_df.plot()
    plt.title('Training Time Curves for Dimensionality Reductions')
    plt.ylabel('Train Time (Seconds)')
    plt.savefig('part5/Wall Times.png')
    plt.close()

    ## Last Wall Time for 200 iterations
    last_df = wall_times_df.iloc[-1:]
    last_df.plot.bar()
    plt.title('Training Times for Clustering')
    plt.ylabel('Train Time in Seconds')
    plt.savefig('part5/Wall Times Bar Plot.png')
    plt.close()


if __name__ == "__main__":
    wine_X, wine_y = clean_wine_data()

    print('===========================Dimensionality Reduction and NN Wine Dataset===========================')
    nn_classifier(wine_X, wine_y)
    # nn(wine_X, wine_y)
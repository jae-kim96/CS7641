import mlrose_hiive as mlrose
from mlrose_hiive import fitness
from mlrose_hiive.neural import activation
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

## Part 2 of the Randomized Optimization HW
## Exploring the weights of a Neural Network

## Using https://mlrose.readthedocs.io/en/stable/source/tutorial3.html#neural-networks

## Wine Data provided from UCI ML Repository
## https://archive.ics.uci.edu/ml/datasets/wine+quality
def clean_wine_data():
    '''
    Reading the Wine data and transforming it into a binary Classification Problem
    Utilizing the Random Over Sampler to balance out the classes.
    '''
    wine_df = pd.read_csv('data/winequality-white.csv', sep = ';')
    wine_df['binary_quality'] = np.where(wine_df['quality'] >= 7, 1, 0)
    
    # print(wine_df['binary_quality'].value_counts())
    X = wine_df.loc[:, 'fixed acidity': 'alcohol']
    y = wine_df.loc[:, 'binary_quality']

    # ros = RandomOverSampler(random_state = 42)
    # X_new, y_new = ros.fit_resample(X, y)
    # print(y.value_counts())
    ## Min Max Normalization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # return X_new, y_new
    return X_scaled, y

def find_params(X, y):
    '''
    Grid Searching for the Best Parameters for MLP Classifier
    '''
    ## Grid Search for best params
    params = {
        'learning_rate_init': [1, 0.1, 0.001, 0.0001],
        # 'hidden_layer_sizes': [(2,), (2, 4), (2, 4, 8), (2, 4, 8, 4), (2, 4, 16, 8)],
        'hidden_layer_sizes': [(32,)(16,), (16, 4), (8, 4), (8, 4, 2), (16, 4, 8, 2), (2, 4)],
        'max_iter': np.arange(100, 1100, 100).tolist(),
        'activation':['identity', 'relu', 'tanh']
    }
    cv = GridSearchCV(MLPClassifier(), params, n_jobs = -1)
    cv.fit(X, y)

    best_param = cv.best_params_
    hidden_layers = best_param['hidden_layer_sizes']
    learning_rate = best_param['learning_rate_init']
    max_iter = best_param['max_iter']
    activation = best_param['activation']

    return hidden_layers, learning_rate, max_iter, activation

def neural_network(X, y, algorithm, hidden_layers, learning_rate, max_iter, activation):
    '''
    A method to observe the behavior of optimizing weights on a neural network
    '''
    ## Training Metrics

    train_times = list()
    fevals = list()
    train_acc = list()
    test_acc = list()
    train_loss = list()
    test_loss = list()
    iteration = list()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # num_iter = max_iter
    num_iters = np.arange(10, 510, 20).tolist()
    for iter in num_iters:
        nn_model = mlrose.NeuralNetwork(
            hidden_nodes = list(hidden_layers),
            activation = activation,
            algorithm = algorithm,
            is_classifier = True,
            curve = True,
            learning_rate = learning_rate, 
            early_stopping = True, 
            max_attempts = 10,
            max_iters = iter, 
            pop_size = 250,
            mutation_prob = 0.25, 
            restarts = 10
        )

        iteration.append(iter)
        ## Training Time
        t0 = time.time()
        nn_model.fit(X_train, y_train)
        train_time = time.time() - t0
        train_times.append(train_time)

        ## Fucntion Evaluations
        feval = nn_model.fitness_curve[-1][0]
        fevals.append(feval)

        ## Predictions
        y_pred_in = nn_model.predict(X_train)
        y_pred_out = nn_model.predict(X_test)

        ## Accuracy
        acc_in = accuracy_score(y_train, y_pred_in)
        acc_out = accuracy_score(y_test, y_pred_out)

        train_acc.append(acc_in)
        test_acc.append(acc_out)
        
        ## Loss
        loss_in = log_loss(y_train, y_pred_in)
        loss_out = log_loss(y_test, y_pred_out)

        train_loss.append(loss_in)
        test_loss.append(loss_out)

    ## Learning Curve
    learning_curve = {
        'Iterations': iteration,
        'Training Accuracy': train_acc,
        'Testing Accuracy': test_acc
    }
    learning_curve_df = pd.DataFrame(learning_curve).set_index('Iterations')
    learning_curve_df.plot()
    plt.title(f'Learning Curve for {algorithm}')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig(f'nn/Learning Curve for {algorithm}.png')

    ## Loss Curve
    loss_curve = {
        'Iterations': iteration,
        'Training Loss': train_loss,
        'Testing Loss': test_loss
    }
    loss_curve_df = pd.DataFrame(loss_curve).set_index('Iterations')
    loss_curve_df.plot()
    plt.title(f'Loss Curve for {algorithm}')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.savefig(f'nn/Loss Curve for {algorithm}.png')

    ## Training Time Curve
    elapsed_time = {
        'Iterations': iteration,
        'Train Times': train_times
    }
    elapsed_time_df = pd.DataFrame(elapsed_time).set_index('Iterations')
    elapsed_time_df.plot()
    plt.title(f'Training Times for {algorithm}')
    plt.xlabel('Iterations')
    plt.ylabel('Time (s)')
    plt.savefig(f'nn/Training Times for {algorithm}.png')

    ## Function Evals Curve
    fevals_plot = {
        'Iterations': iteration,
        'FEvals': fevals
    }
    fevals_plot_df = pd.DataFrame(fevals_plot).set_index('Iterations')
    fevals_plot_df.plot()
    plt.title(f'FEvals for {algorithm}')
    plt.xlabel('Iterations')
    plt.ylabel('FEvals')
    plt.savefig(f'nn/FEvals for {algorithm}.png')


def nn_comparisons(X, y, algorithm, hidden_layers, learning_rate, max_iter, activation):
    '''
    A method for getting the accuracies from each run
    Also provides fitness curve
    '''
    nn_model = mlrose.NeuralNetwork(
        hidden_nodes = list(hidden_layers),
        activation = activation,
        algorithm = algorithm,
        is_classifier = True,
        curve = True,
        learning_rate = learning_rate, 
        early_stopping = True, 
        # max_attempts = 10,
        max_iters = max_iter, 
        pop_size = 250,
        mutation_prob = 0.25
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.4, random_state = 42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)
    ## Fitting Data and Wall Time
    t0 = time.time()
    nn_model.fit(X_train, y_train)
    train_time = time.time() - t0

    ## Return Function Evals
    if algorithm != 'gradient_descent':
        total_fevals = nn_model.fitness_curve[-1][1]

        fitness = [x[0] for x in nn_model.fitness_curve]
        fevals = [x[1] for x in nn_model.fitness_curve]
        iterations = np.arange(1, len(fevals) + 1, 1).tolist()
        
        df = pd.DataFrame({'iterations': iterations, 'fitness': fitness}).set_index('iterations')
        df.plot()
        plt.title(f'Fitness Curve for {algorithm}')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.savefig(f'nn/Fit Curve for {algorithm}.png')

        df2 = pd.DataFrame({'iterations': iterations, 'fevals': fevals}).set_index('iterations')
        df2.plot()
        plt.title(f'Fitness Curve for {algorithm}')
        plt.xlabel('Iterations')
        plt.ylabel('FEvals')
        plt.savefig(f'nn/FEvals Curve for {algorithm}.png')

    else:
        total_fevals = 0
    
    
    y_pred_train = nn_model.predict(X_train)
    y_pred_test = nn_model.predict(X_test)
    y_pred_val = nn_model.predict(X_val)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_val, y_pred_val)

    return train_acc, test_acc, train_acc, train_time, total_fevals


if __name__ == "__main__":
    X, y = clean_wine_data()

    ## GridSearching
    # print('==================FINDING BEST HYPER PARAMETERS==================')
    # hidden_layers, learning_rate, max_iter, act = find_params(X, y)

    # print(f'Hidden Layers = {hidden_layers}')
    # print(f'Learning Rate = {learning_rate}')
    # print(f'Max Iterations = {max_iter}')
    # print(f'Activation = {act}')
    # print('==================DONE BEST HYPER PARAMETERS==================')

    ## From GridSearchCV Best Parameters
    # hidden_layers = (2, 4)
    # learning_rate = 0.1
    # max_iter = 400
    # act = 'tanh'

    # From GridSearchCV Best Parameters
    hidden_layers = (32,)
    learning_rate = 0.1
    max_iter = 500
    act = 'tanh'

    ## Generating Loss and Learning Curves
    algos = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
    for a in algos:
        print(f'=========================RUNNING FOR {a}=========================')
        neural_network(X, y, a, hidden_layers, learning_rate, max_iter, act)

    ## Gathering Metrics
    algos = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']
    for a in algos:
        print(f'=========================RUNNING FOR {a}=========================')
        train_acc, test_acc, val_acc, wall_time, total_fevals = nn_comparisons(X, y, a, hidden_layers, learning_rate, max_iter, act)
        print(f'{a}: Wall Time = {wall_time}, FEvals = {total_fevals}')
        print(f'{a}: Train Acc = {train_acc}, Test Acc = {test_acc}, Val Acc = {val_acc}')

    
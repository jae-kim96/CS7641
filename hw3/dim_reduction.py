from os import error
import pandas as pd
import numpy as np
import time
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error

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

## Function to Clean the Water Potability Data
def clean_water_data():
    water_df = pd.read_csv(water_path).dropna().reset_index(drop = True)
    
    X = water_df.loc[:, 'ph': 'Turbidity']
    y = water_df.loc[:, 'Potability']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(y.value_counts())
    return X_scaled, y

def run_PCA(X, y, dataset_name):
    '''
    Principal Component Analysis for Dimensionality Reduction
    '''
    n_components = list()
    cum_explained_variance = list()
    explained_variance = list()
    times = list()
    mse_list = list()

    for n in range(2, X.shape[1] + 1):
        pca = PCA(n_components = n)
        ## Training Time
        t0 = time.time()
        pca.fit(X)
        train_time = time.time() - t0
        times.append(train_time)

        new_X = pca.fit_transform(X)
        old_X = pca.inverse_transform(new_X)        
        
        ## Metrics
        n_components.append(n)
        cum_explained_variance.append(sum(pca.explained_variance_ratio_))
        explained_variance.append(pca.explained_variance_ratio_[-1])
        mse_list.append(mean_squared_error(X, old_X))

    ## Cumulative Explained Variance
    pca_df = pd.DataFrame({'n_components': n_components, 'cum_explained_variance': cum_explained_variance, 'explained_variance': explained_variance}).set_index('n_components')
    pca_df.plot()
    plt.xlabel('n_components')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance vs Num Components')
    plt.savefig(f'part2/pca/{dataset_name} - Explained Variance.png')

    ## MSE of Reconstruction
    mse_df = pd.DataFrame({'n_components': n_components, 'mse': mse_list}).set_index('n_components')
    mse_df.plot()
    plt.xlabel('n_components')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs n_components')
    plt.savefig(f'part2/pca/{dataset_name} - MSE.png')

    return times, mse_list

def run_ICA(X, y, dataset_name):
    '''
    Independent Component Analysis for Dim Reduction
    '''
    n_components = list()
    kurt = list()
    mse_list = list()
    times = list()

    for n in range(2, X.shape[1] + 1):
        ica = FastICA(n_components = n, max_iter = 500)

        ## Training Time
        t0 = time.time()
        ica.fit(X)
        train_time = time.time() - t0
        times.append(train_time)

        new_X = ica.fit_transform(X)
        old_X = ica.inverse_transform(new_X)

        n_components.append(n)
        kurt.append(np.mean(kurtosis(old_X)))
        mse_list.append(mean_squared_error(X, old_X))
    # print(kurt)
    kurt_df = pd.DataFrame({'n_components': n_components, 'kurtosis': kurt}).set_index('n_components')
    kurt_df.plot()
    plt.xlabel('n_components')
    plt.ylabel('Kurtosis')
    plt.title('Kurtosis vs Num Components')
    plt.savefig(f'part2/ica/{dataset_name} - Kurtosis.png')

    mse_df = pd.DataFrame({'n_components': n_components, 'mse': mse_list}).set_index('n_components')
    mse_df.plot()
    plt.xlabel('n_components')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs n_components')
    plt.savefig(f'part2/ica/{dataset_name} - MSE.png')

    return times, mse_list

def run_RP(X, y, dataset_name):
    '''
    Randomized Projection for Dim Reduction
    '''
    n_components = list()
    mse_list = list()
    times = list()

    for n in range(2, X.shape[1] + 1):
        rp = GaussianRandomProjection(n_components = n)
        
        ## Training Time
        t0 = time.time()
        rp.fit(X)
        train_time = time.time() - t0
        times.append(train_time)

        X_new = rp.fit_transform(X)
        X_old = np.dot(X_new, rp.components_)

        ## Metrics
        n_components.append(n)
        mse_list.append(mean_squared_error(X, X_old))

    ## Plotting MSE
    mse_df = pd.DataFrame({'n_components': n_components, 'mse': mse_list}).set_index('n_components')
    mse_df.plot()
    plt.xlabel('n_components')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs n_components')
    plt.savefig(f'part2/rp/{dataset_name} - MSE.png')

    return times, mse_list

def run_KPCA(X, y, dataset_name):
    '''
    Fourth algorithm for dim reduction. Using Kernel PCA
    '''
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    mse = {}
    times = list()
    for k in kernels:
        # n_components = list()
        
        for n in range(2, X.shape[1] + 1):
            kpca = KernelPCA(kernel = k, n_components = n, fit_inverse_transform = True)
            ## Training Time
            if k == 'poly':
                t0 = time.time()
                kpca.fit(X)
                train_time = time.time() - t0
                times.append(train_time)

            X_new = kpca.fit_transform(X)
            X_old = kpca.inverse_transform(X_new)

            ## Metrics
            if k not in mse.keys():
                mse[k] = [mean_squared_error(X, X_old)]
            else:
                mse[k].append(mean_squared_error(X, X_old))

    # print(mse)
    n_components = np.arange(2, X.shape[1] + 1, 1).tolist()
    mse['n_components'] = n_components
    mse_df = pd.DataFrame(mse).set_index('n_components')
    mse_df.plot()
    plt.xlabel('n_components')
    plt.ylabel('Mean Squared Error')
    plt.title(f'MSE vs n_components for Kernels')
    plt.savefig(f'part2/kpca/{dataset_name} for Kernels - MSE.png')

    return times, mse['poly']


if __name__ == "__main__":
    wine_X, wine_y = clean_wine_data()
    water_X, water_y = clean_water_data()
    print(type(wine_X))
    print(type(wine_y))

    print('===========================Starting Dimensionality Reduction on Wine Dataset===========================')
    wine_pca_time, wine_pca_mse = run_PCA(wine_X, wine_y, 'white-wine-quality')
    wine_ica_time, wine_ica_mse = run_ICA(wine_X, wine_y, 'white-wine-quality')
    wine_rp_time, wine_rp_mse = run_RP(wine_X, wine_y, 'white-wine-quality')
    wine_kpca_time, wine_kpca_mse = run_KPCA(wine_X, wine_y, 'white-wine-quality')

    n_components = np.arange(2, wine_X.shape[1] + 1, 1).tolist()
    wine_times_dict = {
        'n_components': n_components,
        'pca_time': wine_pca_time,
        'ica_time': wine_ica_time,
        'rp_time': wine_rp_time,
        # 'kpca_time': wine_kpca_time
    }
    ## Wall Times
    times_df = pd.DataFrame(wine_times_dict).set_index('n_components')
    times_df.plot()
    plt.title('Time to Fit for Dim Reduction')
    plt.xlabel('n_components')
    plt.ylabel('Fit Time')
    plt.savefig('part2/Fit Times for Wine Dataset.png')
    plt.close()

    ## MSE List
    wine_mse_dict= {
        'n_components': n_components,
        'pca_mse': wine_pca_mse,
        'ica_mse': wine_ica_mse,
        'rp_mse': wine_rp_mse,
        'kpca_mse': wine_kpca_mse
    }
    mse_df = pd.DataFrame(wine_mse_dict).set_index('n_components')
    mse_df.plot()
    plt.title('MSE for Dim Reduction')
    plt.xlabel('n_components')
    plt.ylabel('MSE')
    plt.savefig('part2/MSE for Wine Dataset.png')
    plt.close()
    
    print('===========================Starting Dimensionality Reduction on Water Dataset===========================')
    water_pca_time, water_pca_mse = run_PCA(water_X, water_y, 'water-potability')
    water_ica_time, water_ica_mse = run_ICA(water_X, water_y, 'water-potability')
    water_rp_time, water_rp_mse = run_RP(water_X, water_y, 'water-potability')
    water_kpca_time, water_kpca_mse = run_KPCA(water_X, water_y, 'water-potability')

    n_components = np.arange(2, water_X.shape[1] + 1, 1).tolist()
    water_times_dict = {
        'n_components': n_components,
        'pca_time': water_pca_time,
        'ica_time': water_ica_time,
        'rp_time': water_rp_time,
        # 'kpca_time': water_kpca_time
    }
    times_df = pd.DataFrame(water_times_dict).set_index('n_components')
    times_df.plot()
    plt.title('Time to Fit for Dim Reduction')
    plt.xlabel('n_components')
    plt.ylabel('Fit Time')
    plt.savefig('part2/Fit Times for Water Dataset.png')

    ## MSE List
    wine_times_dict = {
        'n_components': n_components,
        'pca_mse': water_pca_mse,
        'ica_mse': water_ica_mse,
        'rp_mse': water_rp_mse,
        'kpca_mse': water_kpca_mse
    }
    mse_df = pd.DataFrame(wine_mse_dict).set_index('n_components')
    mse_df.plot()
    plt.title('MSE for Dim Reduction')
    plt.xlabel('n_components')
    plt.ylabel('MSE')
    plt.savefig('part2/MSE for Water Dataset.png')
    plt.close()
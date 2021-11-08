import pandas as pd
import numpy as np
import time
import warnings
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, rand_score, homogeneity_score, accuracy_score, calinski_harabasz_score

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

def plot_results(result_dict, reduction_tech, dataset_name):
    '''
    Creating plots for the results
    '''
    for c in result_dict.keys():
        print(f'-->Plotting {c}')
        curr = result_dict[c]
        if c == 'kmeans':
            inertia_df = pd.DataFrame({'n_clusters': curr['n_clusters'], 'inertia': curr['inertia']}).set_index('n_clusters')
            inertia_df.plot()
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title(f'{reduction_tech}-KMeans-{dataset_name} Elbow Curve with Inertia')
            plt.savefig(f'part3/{reduction_tech}/{c}/{dataset_name}-{c}-{reduction_tech} - Elbow Curve.png')
            plt.close()
        else:
            bic_df = pd.DataFrame({'n_clusters': curr['n_clusters'], 'bic': curr['bic']}).set_index('n_clusters')
            bic_df.plot()
            plt.xlabel('Number of Clusters')
            plt.ylabel('BIC')
            plt.title(f'{reduction_tech}-GMM-{dataset_name} BIC vs Num Clusters')
            plt.savefig(f'part3/{reduction_tech}/{c}/{dataset_name}-{c}-{reduction_tech} - BIC Curve.png')
            plt.close()

        sil_df = pd.DataFrame({'n_clusters': curr['n_clusters'], 'Silhouette Score': curr['sil_scores']}).set_index('n_clusters')
        sil_df.plot()
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title(f'{reduction_tech}-{c}-{dataset_name} Silhouette Scores vs N_Clusters')
        plt.savefig(f'part3/{reduction_tech}/{c}/{dataset_name}-{c}-{reduction_tech} - Silhouette Scores.png')
        plt.close()

        cal_df = pd.DataFrame({'n_clusters': curr['n_clusters'], 'Calinski Harabaz Score': curr['cal_scores']}).set_index('n_clusters')
        cal_df.plot()
        plt.xlabel('Number of Clusters')
        plt.ylabel('Calinski Harabaz Score')
        plt.title(f'{reduction_tech}-{c}-{dataset_name}Calinski Harabaz Score vs N_Clusters')
        plt.savefig(f'part3/{reduction_tech}/{c}/{dataset_name}-{c}-{reduction_tech} - CH Scores.png')
        plt.close()

        homo_df = pd.DataFrame({'n_clusters': curr['n_clusters'], 'Homogeneity': curr['homo_scores']}).set_index('n_clusters')
        homo_df.plot()
        plt.xlabel('Number of Clusters')
        plt.ylabel('Homogeneity')
        plt.title(f'{reduction_tech}-{c}-{dataset_name} Homogeneity vs N_Clusters')
        plt.savefig(f'part3/{reduction_tech}/{c}/{dataset_name}-{c}-{reduction_tech} - Homogeneity.png')
        plt.close()

## Wine Dataset Experiment
def wine_dim_reduction_clustering(X, y, dataset_name):
    '''
    Dimensionality Reduction and CLustering on the Wine Dataset
    '''

    ## Dimensionality Reduction Fit from Part 2
    dim_reductions = {
        'pca': PCA(n_components = 4),
        'ica': FastICA(n_components = 11, max_iter = 800),
        'rp': GaussianRandomProjection(n_components = 8),
        'kpca': KernelPCA(n_components = 6, kernel = 'poly')
    }

    ## Going through reduction algorithms
    for tech in dim_reductions.keys():
        curr_dim  = dim_reductions[tech]
        X_red = curr_dim.fit_transform(X)

        ## Keeping track of results for each CLustering Algorithm
        clustering_results = {
            'kmeans': {
                'n_clusters': list(),
                'inertia': list(),
                'sil_scores': list(),
                'cal_scores': list(),
                'homo_scores': list()
            },
            'gmm': {
                'n_clusters': list(),
                'sil_scores': list(),
                'cal_scores': list(),
                'homo_scores': list(),
                'bic': list()
            }
        }

        print(f'Running {tech} and KMeans CLustering')
        ## Running KMeans ALgorithm
        for n in range(2, (X.shape[1]) * 5):
            km = KMeans(n_clusters = n)
            km.fit(X_red)
            pred = km.predict(X_red)

            ## Adding Metrics
            clustering_results['kmeans']['n_clusters'].append(n)
            clustering_results['kmeans']['inertia'].append(km.inertia_)
            clustering_results['kmeans']['sil_scores'].append(silhouette_score(X_red, pred, metric = 'euclidean'))
            clustering_results['kmeans']['cal_scores'].append(calinski_harabasz_score(X_red, pred))
            clustering_results['kmeans']['homo_scores'].append(homogeneity_score(y, pred))
            
        print(f'Running {tech} and GMM CLustering')
        ## Running GMM ALgorithm
        for n in range(2, (X.shape[1]) * 5):
            gmm = GaussianMixture(n_components = n)
            gmm.fit(X_red)
            pred = gmm.predict(X_red)

            ## Adding Metrics
            clustering_results['gmm']['n_clusters'].append(n)
            clustering_results['gmm']['sil_scores'].append(silhouette_score(X_red, pred, metric = 'euclidean'))
            clustering_results['gmm']['cal_scores'].append(calinski_harabasz_score(X_red, pred))
            clustering_results['gmm']['homo_scores'].append(homogeneity_score(y, pred))
            clustering_results['gmm']['bic'].append(gmm.bic(X_red))


        plot_results(clustering_results, tech, dataset_name)

## Water Dataset Experiment
def water_dim_reduction_clustering(X, y, dataset_name):
    '''
    Dimensionality Reduction and CLustering on the Wine Dataset
    '''

    ## Dimensionality Reduction Fit from Part 2
    dim_reductions = {
        'pca': PCA(n_components = 7),
        'ica': FastICA(n_components = 9, max_iter = 800),
        'rp': GaussianRandomProjection(n_components = 5),
        'kpca': KernelPCA(n_components = 7, kernel = 'poly')
    }

    ## Going through reduction algorithms
    for tech in dim_reductions.keys():
        curr_dim  = dim_reductions[tech]
        X_red = curr_dim.fit_transform(X)

        ## Keeping track of results for each CLustering Algorithm
        clustering_results = {
            'kmeans': {
                'n_clusters': list(),
                'inertia': list(),
                'sil_scores': list(),
                'cal_scores': list(),
                'homo_scores': list()
            },
            'gmm': {
                'n_clusters': list(),
                'sil_scores': list(),
                'cal_scores': list(),
                'homo_scores': list(),
                'bic': list()
            }
        }

        print(f'Running {tech} and KMeans CLustering')
        ## Running KMeans ALgorithm
        for n in range(2, (X.shape[1]) * 5):
            km = KMeans(n_clusters = n)
            km.fit(X_red)
            pred = km.predict(X_red)

            ## Adding Metrics
            clustering_results['kmeans']['n_clusters'].append(n)
            clustering_results['kmeans']['inertia'].append(km.inertia_)
            clustering_results['kmeans']['sil_scores'].append(silhouette_score(X_red, pred, metric = 'euclidean'))
            clustering_results['kmeans']['cal_scores'].append(calinski_harabasz_score(X_red, pred))
            clustering_results['kmeans']['homo_scores'].append(homogeneity_score(y, pred))
            
        print(f'Running {tech} and GMM CLustering')
        ## Running GMM ALgorithm
        for n in range(2, (X.shape[1]) * 5):
            gmm = GaussianMixture(n_components = n)
            gmm.fit(X_red)
            pred = gmm.predict(X_red)

            ## Adding Metrics
            clustering_results['gmm']['n_clusters'].append(n)
            clustering_results['gmm']['sil_scores'].append(silhouette_score(X_red, pred, metric = 'euclidean'))
            clustering_results['gmm']['cal_scores'].append(calinski_harabasz_score(X_red, pred))
            clustering_results['gmm']['homo_scores'].append(homogeneity_score(y, pred))
            clustering_results['gmm']['bic'].append(gmm.bic(X_red))


        plot_results(clustering_results, tech, dataset_name)



if __name__ == "__main__":
    wine_X, wine_y = clean_wine_data()
    water_X, water_y = clean_water_data()

    print('===========================Starting Dimensionality Reduction and Clustering on Wine Dataset===========================')
    wine_dim_reduction_clustering(wine_X, wine_y, 'white-wine-quality')

    print('===========================Starting Dimensionality Reduction and Clustering on Water Dataset===========================')
    water_dim_reduction_clustering(water_X, water_y, 'water-potability')
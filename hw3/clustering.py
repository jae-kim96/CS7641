import pandas as pd
import numpy as np
import time
import warnings

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

## Initial Testing of the Clustering Algorithms. Testing
## 1) Silhouette Score
## 2) Rand Index
## 3) Calinski Harabasz Index
## 4) Homogeneity Score
def KMeansClustering(X, y, dataset_name):
    '''
    Experiments for looking at clustering of the datasets
    '''
    inertia = list()
    sil_scores = list()
    rand_inds = list()
    cal_scores = list()
    homo_scores = list()
    # accuracy_scores = list()

    # n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_clusters = list()

    for n in range(2, 41):
        n_clusters.append(n)
        km = KMeans(n_clusters = n)
        km.fit(X)
        pred = km.predict(X)

        ## Adding Metrics
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X, pred, metric = 'euclidean'))
        rand_inds.append(rand_score(y, pred))
        cal_scores.append(calinski_harabasz_score(X, pred))
        homo_scores.append(homogeneity_score(y, pred))
        # accuracy_scores.append(accuracy_score(y, pred))


    ## Elbow Curve using Inertia compared to n_clusters
    inertia_df = pd.DataFrame({'n_clusters': n_clusters, 'inertia': inertia}).set_index('n_clusters')
    inertia_df.plot()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve for KMeans with Inertia')
    plt.savefig(f'part1/kmeans/{dataset_name} - Elbow Curve (Inertia).png')
    plt.close()

    ## Silhouette Score
    sil_df = pd.DataFrame({'n_clusters': n_clusters, 'Silhouette Score': sil_scores}).set_index('n_clusters')
    sil_df.plot()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores vs N_Clusters')
    plt.savefig(f'part1/kmeans/{dataset_name} - Silhouette Scores.png')
    plt.close()

    ## Rand Score
    rand_df = pd.DataFrame({'n_clusters': n_clusters, 'Rand Index': rand_inds}).set_index('n_clusters')
    rand_df.plot()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Rand Index')
    plt.title('Rand Index vs N_Clusters')
    plt.savefig(f'part1/kmeans/{dataset_name} - Rand Index.png')
    plt.close()

    ## Calinski Harabaz Score
    cal_df = pd.DataFrame({'n_clusters': n_clusters, 'Calinski Harabaz Score': cal_scores}).set_index('n_clusters')
    cal_df.plot()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski Harabaz Score')
    plt.title('Calinski Harabaz Score vs N_Clusters')
    plt.savefig(f'part1/kmeans/{dataset_name} - Calinski Harabaz Scores.png')
    plt.close()

    ## Homogeneity Score
    homo_df = pd.DataFrame({'n_clusters': n_clusters, 'Homogeneity': homo_scores}).set_index('n_clusters')
    homo_df.plot()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Homogeneity')
    plt.title('Homogeneity vs N_Clusters')
    plt.savefig(f'part1/kmeans/{dataset_name} - Homogeneity.png')
    plt.close()

    pass


def GaussianMixClustering(X, y, dataset_name):
    '''
    Experiments for Expectation Maximization on datasets
    '''
    sil_scores = list()
    rand_inds = list()
    cal_scores = list()
    homo_scores = list()
    aic = list()
    bic = list()
    # accuracy_scores = list()

    # n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_components = list()
    for n in range(2, 41):
        n_components.append(n)
        gmm = GaussianMixture(n_components = n)
        gmm.fit(X)
        pred = gmm.predict(X)

        ## Adding Metrics
        sil_scores.append(silhouette_score(X, pred, metric = 'euclidean'))
        rand_inds.append(rand_score(y, pred))
        cal_scores.append(calinski_harabasz_score(X, pred))
        homo_scores.append(homogeneity_score(y, pred))
        aic.append(gmm.aic(X))
        bic.append(gmm.bic(X))
        # accuracy_scores.append(accuracy_score(y, pred))

    ## Silhouette Score
    sil_df = pd.DataFrame({'n_components': n_components, 'Silhouette Score': sil_scores}).set_index('n_components')
    sil_df.plot()
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores vs n_components')
    plt.savefig(f'part1/gmm/{dataset_name} - Silhouette Scores.png')
    plt.close()

    ## Rand Score
    rand_df = pd.DataFrame({'n_components': n_components, 'Rand Index': rand_inds}).set_index('n_components')
    rand_df.plot()
    plt.xlabel('Number of Components')
    plt.ylabel('Rand Index')
    plt.title('Rand Index vs n_components')
    plt.savefig(f'part1/gmm/{dataset_name} - Rand Index.png')
    plt.close()

    ## Calinski Harabaz Score
    cal_df = pd.DataFrame({'n_components': n_components, 'Calinski Harabaz Score': cal_scores}).set_index('n_components')
    cal_df.plot()
    plt.xlabel('Number of Components')
    plt.ylabel('Calinski Harabaz Score')
    plt.title('Calinski Harabaz Score vs n_components')
    plt.savefig(f'part1/gmm/{dataset_name} - Calinski Harabaz Scores.png')
    plt.close()

    ## Homogeneity Score
    homo_df = pd.DataFrame({'n_components': n_components, 'Homogeneity': homo_scores}).set_index('n_components')
    homo_df.plot()
    plt.xlabel('Number of Components')
    plt.ylabel('Homogeneity')
    plt.title('Homogeneity vs n_components')
    plt.savefig(f'part1/gmm/{dataset_name} - Homogeneity.png')
    plt.close()

    ## AIC Score
    aic_df = pd.DataFrame({'n_components': n_components, 'AIC': aic}).set_index('n_components')
    aic_df.plot()
    plt.xlabel('Number of Components')
    plt.ylabel('AIC')
    plt.title('AIC vs n_components')
    plt.savefig(f'part1/gmm/{dataset_name} - AIC.png')
    plt.close()

    ## BIC Score
    bic_df = pd.DataFrame({'n_components': n_components, 'BIC': bic}).set_index('n_components')
    bic_df.plot()
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.title('BIC vs n_components')
    plt.savefig(f'part1/gmm/{dataset_name} - BIC.png')
    plt.close()
    pass


if __name__ == "__main__":
    wine_X, wine_y = clean_wine_data()
    water_X, water_y = clean_water_data()

    ## KMeans Clustering
    print('===========================Starting Analysis on KMeans Clustering===========================')
    KMeansClustering(wine_X, wine_y, 'white-wine-quality')
    KMeansClustering(water_X, water_y, 'water-potability')

    print('===========================Starting Analysis on GMM Clustering===========================')
    GaussianMixClustering(wine_X, wine_y, 'white-wine-quality')
    GaussianMixClustering(water_X, water_y, 'water-potability')
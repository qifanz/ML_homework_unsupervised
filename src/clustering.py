import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from keras.utils import to_categorical

from src.reduce_dimensions import get_data2


def clustering(dataset, algorithm: str, n_clusters: int, out_file_name: str =None):

    if type(dataset) == str:
        df = pd.read_csv(dataset, ',', header=None)
    elif type(dataset) == pd.DataFrame:
        df = dataset
    else:
        try:
            df = pd.DataFrame(dataset)
        except ValueError:
            raise TypeError("dataset must be a pandas dataframe or a path to a csv file")

    if algorithm == "kmeans":
        algo = KMeans(n_clusters=n_clusters,
                      n_init=30,
                      n_jobs=-1
                      )
    elif algorithm == "EM":
        algo = GaussianMixture(n_components=n_clusters,
                               n_init=30,
                               init_params='kmeans'  # can also be kmeans or random !
                               )
    else:
        raise ValueError("algorithm must be one of 'kmeans' or 'EM'")

    algo.fit(X=df)
    if algorithm == 'kmeans':
        print(algo.inertia_)

    p = algo.predict(X=df)

    dfp = df.assign(label=p)

    if out_file_name is not None:
        dfp.to_csv(out_file_name, sep=',', encoding='utf-8', header=None)

    return dfp


def get_clustered_data(number_of_feature: int, algorithm: str, clustering_name: str, number_of_classes: int):
    x_train, y_train, x_test, y_test = get_data2(number_of_feature, algorithm)

    x_train, algo = clustering(x_train, clustering_name, number_of_classes)

    labels = x_train.get('label')
    x_train = x_train.drop('label', 1)
    labels = to_categorical(labels, number_of_classes)
    x_train = x_train.join(pd.DataFrame(labels), rsuffix="_label")

    labels = algo.predict(x_test)
    labels = to_categorical(labels, number_of_classes)
    x_test = pd.DataFrame(x_test)
    x_test = x_test.join(pd.DataFrame(labels), rsuffix="_label")

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    iris = load_iris()
    clustering(dataset=iris['data'], algorithm='kmeans', n_clusters=3, out_file_name="results/kmeans_iris.csv")






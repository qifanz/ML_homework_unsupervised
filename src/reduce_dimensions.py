import numpy as np
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from src.utils import *


def reduce_dimension(dataset, algorithm: str, nb_of_dimention: int, out_file=None, plot_curve=False, labels=None,title="",ica_sort=True):

    if type(dataset) == str:
        df = pd.read_csv(dataset, ',', header=None)
    elif type(dataset) == pd.DataFrame:
        df = dataset
    else:
        try:
            df = pd.DataFrame(dataset)
        except ValueError:
            raise TypeError("dataset must be a pandas dataframe or a path to a csv file")

    if algorithm == "PCA":
        algo = PCA(n_components=nb_of_dimention,
                   svd_solver='full')
    elif algorithm == 'ICA':
        algo = FastICA(n_components=nb_of_dimention)
    elif algorithm == 'random':
        algo = GaussianRandomProjection(n_components=nb_of_dimention)
    elif algorithm == 'FA':
        algo = FactorAnalysis(n_components=nb_of_dimention)


        if labels is None:
            raise ValueError("Labels must be given when using LDA algorithm (but are use less in other case)")
    else:
        raise ValueError("Unknow algorithm, please use one of: 'PCA', 'ICA', 'LDA' or 'random'")

    algo.fit(X=df,
             y=labels)

    tdf = algo.transform(X=df)
    tdf = pd.DataFrame(tdf, columns=list(map(str, range(nb_of_dimention))))

    if out_file is not None:
        tdf.to_csv(out_file, sep=',', encoding='utf-8')

    if plot_curve:
        if algorithm == 'PCA':
            eigenvalues = algo.explained_variance_
            score_list = np.zeros((2, len(eigenvalues)), dtype=np.float64)
            old_sum = 0.0
            for i, v in enumerate(eigenvalues):
                old_sum += v
                score_list[0, i] = old_sum
                score_list[1, i] = v
            plt.figure(1)
            plt.plot(range(1,len(eigenvalues)+1), score_list[0] / np.max(score_list[0]), label="cumulative variance")
            plt.plot(range(1,len(eigenvalues)+1), score_list[1] / np.max(score_list[1]), label="variance")
            plt.legend()
            plt.title(title)
            plt.xlabel("number of dimensions")
            plt.savefig("graphs/" + algorithm + "_dimension_importance_" + str(nb_of_dimention) + ".png")
            plt.show()
        elif algorithm == 'ICA':
            res=[]
            for nb in range(nb_of_dimention):
                ## print(kurtosis(tdf.get(str(nb)), fisher=False))
                res.append(kurtosis(tdf.get(str(nb)), fisher=False))
            if (ica_sort):
                res.sort(reverse = True)
            plt.figure(1)
            plt.plot(range(1,len(res)+1),res)
            plt.legend()
            plt.title(title)

            plt.xlabel("# of Dimension")
            plt.ylabel("kurtosis")
            plt.savefig("graphs/" + algorithm + "_dimension_importance_" + str(nb_of_dimention) + ".png")
            plt.show()


        else:
            raise ValueError('Curve not yet implemented for other algorithm than PCA')
    return tdf, algo


def get_data(number_of_feature: int, algorithm: str):
    TRAIN_DATASET_LENGHT = 21000
    basketball = load_basketball_to_df()
    label = basketball.get('label')
    basketball = basketball.drop('label', 1)
    if (algorithm!='None'):
        data, algo = reduce_dimension(basketball[:TRAIN_DATASET_LENGHT],
                                      algorithm,
                                      number_of_feature,
                                      labels=label[:TRAIN_DATASET_LENGHT]
                                      )
    x_train = data[:TRAIN_DATASET_LENGHT]
    y_train = label[:TRAIN_DATASET_LENGHT]

    x_test = data[TRAIN_DATASET_LENGHT:]
    y_test = label[TRAIN_DATASET_LENGHT:]
    return x_train, y_train, x_test, y_test

def get_data2(number_of_feature: int, algorithm: str):
    df = pd.read_csv("data/basketball.csv", header=0)
    df = df.dropna()
    df = pd.get_dummies(df)
    print(df.shape)
    data = df.drop('label', axis=1).values
    target = df.get('label').values
    target = target.reshape((len(target), 1))
    if (algorithm != 'None'):
        data, algo = reduce_dimension(data,
                                      algorithm,
                                      number_of_feature,
                                      labels=target
                                      )
    else:
        pass
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.25,
                                                        shuffle=True)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # iris = load_iris_to_df()
    # iris = iris.drop('label', 1)
    # pca_iris = reduce_dimension(iris, 'PCA', 4, plot_curve=True)
    # ica_iris = reduce_dimension(iris, 'ICA', 2)
    '''
    sc = load_starcraft_to_df()
    labels = sc.get('label')
    sc = sc.drop('label', 1)
    print(labels.shape, sc.shape)
    pca_sc = reduce_dimension(sc, 'LDA', 70, plot_curve=True, labels=labels)'''

    iris = load_iris_to_df()
    iris_label = iris.get('label')
    iris = iris.drop('label', 1)
    ica_iris_1, _ = reduce_dimension(iris, 'LDA',4,plot_curve=True)



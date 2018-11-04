import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def load_basketball_to_df():
    basketball=pd.read_csv("data/basketball.csv")
    basketball=basketball.dropna()
    basketball=pd.get_dummies(basketball)
    return basketball


def load_iris_to_df():
    iris = load_iris()
    df = pd.DataFrame(iris['data'], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df = df.assign(label=iris['target'])
    return df


def load_starcraft_to_df():
    sc = pd.read_csv("data/starcraft_x_all.csv", header=0)
    # normalize data set
    min_sc, max_sc = np.min(sc, 0), np.max(sc, 0)
    sc[:] -= min_sc
    sc[:] /= max_sc - min_sc
    # add label
    sc_lb = pd.read_csv("data/starcraft_nl_y_all.csv", header=0)
    sc = sc.assign(label=sc_lb['label'])
    return sc

def load_letters_to_df():
    letters=pd.read_csv("data/letter-recognition.csv")
    letters['label']=letters['label'].astype('category')
    letters['label']=letters['label'].cat.codes

    ##min_letter, max_letter = np.min(letters, 0), np.max(letters, 0)
    ##letters[:] -= min_letter
    ##letters[:] /= max_letter - min_letter
    return letters
def load_digits_to_df():
    letters = pd.read_csv("data/digits.csv")
    letters['label'] = letters['label'].astype('category')
    letters['label'] = letters['label'].cat.codes

    ##min_letter, max_letter = np.min(letters, 0), np.max(letters, 0)
    ##letters[:] -= min_letter
    ##letters[:] /= max_letter - min_letter
    return letters

def plot_2d(df: pd.DataFrame, x_axis: str, y_axis: str, label_axis: str='label', comment="",title=""):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.scatter(df.get(x_axis), df.get(y_axis), c=df.get(label_axis), cmap='gist_ncar', linewidths=1, edgecolors='black')
    ax.set_xlabel(x_axis.replace('_', ' '), fontsize=15)
    ax.set_ylabel(y_axis.replace('_', ' '), fontsize=15)
    if comment != "":
        comment += "_"
    plt.savefig("graphs/" + comment + x_axis + "_" + y_axis + "_" + label_axis + ".png")
    plt.show()


def plot_3d(df: pd.DataFrame, x_axis: str, y_axis: str, z_axis: str, label_axis: str='label', comment=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.get(x_axis), df.get(y_axis), df.get(z_axis), c=df.get(label_axis),
               s=5, depthshade=False, cmap='gist_ncar')
    ax.set_xlabel(x_axis.replace('_', ' '), fontsize=10)
    ax.set_ylabel(y_axis.replace('_', ' '), fontsize=10)
    ax.set_zlabel(z_axis.replace('_', ' '), fontsize=10)
    if comment != "":
        comment += "_"
    plt.savefig("graphs/3D_" + comment + x_axis + "_" + y_axis + "_" + z_axis + "_" + label_axis + ".png")
    plt.show()




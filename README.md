# Machine Learning Homework 3: Unsupervised Learning and Dimensionality Reduction

## Dependency

 - Python 
 - Keras 
 - sklearn 
 - jupyter
 - sqlite3
 

## Dataset

All the data ued in this work are provided on
the `data` directory.

## Running the experiments

To get the same results as on the report you just 
have to run the python scripts in the `src` directory.

To run the perceptron with dimensionality reduction:
```bash
python3 -m src.perceptron
```

To run the perceptron with dimention reductionality
and add clustering results as features:
```bash
python3 -m src.clustered_perceptron
```

Both of them will generate `.csv` files in the 
`stats` directory. This to files need to be loaded
into the a sqlite3 database named `data.db` 
in order to be used correctly by the different jupyter
notebook.

You will find more information on the database
schema on first cell of the different notebooks. 

## Plotting the graphs

The four part of this work are divided on four
different notebooks:

 - Clustering: `plotter/dataset.ipynb`
 - Dimension reduction `plotter/dimesion_reduction.ipynb`
 - Perceptron with dimension reduction `plotter/nn.ipynb`
 - Perceptron with dimension reduction and clustering `plotter/clusper.ipynb`


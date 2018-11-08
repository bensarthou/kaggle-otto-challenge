# kaggle-otto-challenge

Challenge Kaggle of Otto Products classification, for course TC1 of master AIC at Université Paris-Sud.

Autors : M. Bauw, N. Cadart, B. Sarthou

Date : November 2018


## Use instructions

- Copy kaggle otto challenge datasets into the `data/` directory. Files can be downloaded [here](https://www.kaggle.com/c/otto-group-product-classification-challenge/data).
- Run `python3 final_training.py` to train selected models, print results and save them to .csv file respecting kaggle submission format.

The execution may last about 10 minutes.

The final results should reach a log-loss score around 0.457.


## Files description

- **`data_exploration.py`** : compute and display information about the dataset : PCA, T-SNE, features importance, correlation matrix, sparsity...
- **`data_preprocessing.py`** : script ot test several ideas on data classification, such as features normalization, features selection, dimension reduction or distribution probabilities calibration.
- **`toolbox.py`** : useful tools functions for data parsing
- **`gridsearch_training.py`**: run a grid search on several classification models with different hyper-parameters and save results to .csv files in directory `gridsearch_results/`.
- **`final_training.py`** : train the selected models, find ensemble weights, print validation set results, run final predictions on kaggle test sets and save them in .csv file.
- **`data/`** : directory where datasets, results or images are saved.
- **`gridsearch_results/`** : directory where grid search results are savec when running `gridsearch_training` file.

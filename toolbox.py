import numpy as np


TRAIN_CSV_PATH = "./data/train.csv"
TEST_CSV_PATH = "./data/test.csv"


def load_otto_db(test=False):
    """ @brief: Load an Otto dataset from csv file.
        @param:
            test: bool, load Train or Test dataset
        @return:
            obs: ndarray, of size (n_samples, 93), dtype = int
            target: ndarray, of size (n_samples, ), dtype = int
    """
    if not test:
        data = np.genfromtxt(TRAIN_CSV_PATH, delimiter=',',  skip_header=1,
                            converters={94: lambda s: float(str(s)[8])})
                            # convert class name from string to int
        obs = data[:, 1:-1].astype(int)
        target = data[:, -1].astype(int)
        return obs, target

    else:
        data = np.genfromtxt(TEST_CSV_PATH, delimiter=',', skip_header=1)
        obs = data[:, 1:].astype(int)
        return obs

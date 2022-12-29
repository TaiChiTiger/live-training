from sklearn.datasets import make_regression, make_classification
import numpy as np
import pandas as pd


class SyntheticData():
    def lin_sep_reg(self, n_samples, noise, outliers_ratio,
                strength=1, outliers_sep=0.5, seed=123):
        X, y, coef = make_regression(n_samples=n_samples, n_features=1, bias=3,
                                    coef=True, n_informative=1, noise=noise, 
                                    random_state=seed)
        # weights = np.r_[bias, coef]
        if outliers_ratio == 0:
            return X, y

        np.random.seed(seed)
        outliers_idx = np.random.choice(range(int(n_samples)), 
                        int(n_samples*outliers_ratio), replace=False)
        for i in outliers_idx:
            
            if np.random.random() > outliers_sep:
                y[i] += strength * y.std()	
            else:
                y[i] -= strength * y.std()
        

        return X_train, X_test, y_train, y_test, weights, outliers_idx

    def nonlin_sep_reg(self, n_samples=200, noise=0.2, outliers_ratio=0,
                strength=1, outliers_sep=0.5, seed=123):
        np.random.seed(seed)
        X = np.random.rand(n_samples) * 10 - 5
        f = lambda x: np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2)
        y = f(X) + np.random.normal(0.0, noise, n_samples)
        X = X.reshape(-1, 1)
        y = y.ravel()

        if outliers_ratio == 0:
            return X, y

        outliers_idx = np.random.choice(range(n_samples), int(n_samples*outliers_ratio), replace=False)
        for i in outliers_idx:
            if np.random.random() > outliers_sep:
                y[i] += strength * y.std()	
            else:
                y[i] -= strength * y.std()
        

        return X, y, outliers_idx


    def lin_sep_2dcls(self, sample_size=200, seed=123):
        np.random.seed(seed)
		# define sample size of each class
        size0 = int(sample_size / 2)
        size1 = sample_size - size0
		# uncorrelation between two classes
        sigma = 1.0
        covar = sigma ** 2 * np.eye(2)
		# generate the dataset
        X1 = np.random.multivariate_normal([-4, -4], covar, size0)
        y1 = np.repeat(0, size0)
        X2 = np.random.multivariate_normal([0, 0], covar, size1)
        y2 = np.repeat(1, size1)
        X = np.r_[X1, X2]
        y = np.r_[y1, y2]
        # make the dataset
        data = np.c_[X, y]
        np.random.shuffle(data)
        X = data[:, :2]
        y = data[:, -1]

        return X, y

    def lin_sep_1dcls(sample, sample_size=100, seed=123):
        np.random.seed(seed)
        size0 = int(sample_size / 2)
        size1 = sample_size - size0
        x0 = np.random.uniform(0, -2, size0)
        x1 = np.random.uniform(0.5, 2, size1)
        X = np.r_[x0, x1]
        y = np.r_[np.zeros(size0), np.ones(size1)]
        data = np.c_[X, y]
        np.random.shuffle(data)
        X = data[:, 0]
        y = data[:, 1]

        return X, y

    def nonlin_sep_2dcls(self, sample_size=200, seed=123):
        X, y = make_classification(sample_size, n_features=2, 
                                n_informative=2, n_redundant=0, 
                                class_sep=0.1, random_state=seed)

        return X, y
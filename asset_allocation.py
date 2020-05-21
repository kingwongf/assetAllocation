import pandas as pd
import numpy as np



class bootstrap:
    def __init__(self, df_y=None,y1=None, y2=None, to_diff=True, n=1, M=1):
        '''
        :param df_y: df of 2 or more time series
        :param y1: first time series
        :param y2: second time series
        :param n: sampling n observations
        :param M: simulation run
        '''

        self.df_y = y1.merge(y2, left_index=True, right_index=True) if isinstance(y1, pd.Series) and isinstance(y2, pd.Series) else None
        self.df_y = df_y if isinstance(df_y, pd.DataFrame) else None
        self.df_y = self.df_y.pct_change(1).dropna(0) if to_diff else self.df_y
        rng = np.random.default_rng()
        self.col_names = self.df_y.columns
        self.sampled_vs = [rng.choice(self.df_y, n) for _ in range(M)]
        # self.sampled_dfs = [self.df_y.pct_change(1).sample(n, replace=True) for _ in range(M)] if to_diff else [self.df_y.sample(n, replace=True) for _ in range(M)]


    def corr(self):
        corrs = np.array([np.corrcoef(v.T) for v in self.sampled_vs])
        return pd.DataFrame(np.mean(corrs, axis=0), columns=self.col_names, index=self.col_names)

    def cov(self):
        covs = np.array([np.cov(v.T) for v in self.sampled_vs])
        return pd.DataFrame(np.mean(covs, axis=0), columns=self.col_names, index=self.col_names)

    def std(self):
        stds = np.array([v.std(axis=0) for v in self.sampled_vs])
        return pd.DataFrame(np.mean(stds, axis=0), columns=['std'], index=self.col_names)

    def mean(self):
        means = np.array([v.mean(axis=0) for v in self.sampled_vs])
        return pd.DataFrame(np.mean(means, axis=0), columns=['mean'], index=self.col_names)

    def exp_var(self, w_arr=None):
        '''

        :param w_arr: array of weights, dim: 1 * m
        :return: wHw'
        '''
        w_arr = 1/len(self.col_names)*np.ones(shape=(1,len(self.col_names))) if w_arr is None else w_arr
        return w_arr.dot(self.cov()).dot(w_arr.transpose())

class momentum:
    def __init__(self,ts):
        self.ts = ts
    def exponential_momentum(self, min_nobs, window):
        '''
        Andrew Clenow's Method
        1. ln(ts) = m*ln(t) + c
        2. annualised momentum = ((e^(m))^(252) -1 ) * 100
        :return:
            annualised momentum score
        '''
        from statsmodels.regression.rolling import RollingOLS
        import statsmodels.api as sm
        exog = sm.add_constant(np.arange(0, len(self.ts)))
        rolling_param = RollingOLS(np.log(self.ts), exog, min_nobs=min_nobs, window=window).fit()
        return (np.power(np.exp(rolling_param.params['x1']), 252)-1)*100 * rolling_param.rsquared

class hierarchical_risk_parity:

    def __init__(self, df=None, corr=None, cov=None):
        '''

        :param df:
        :param corr: must needed
        :param cov: must needed
        '''

        self.df = df
        corr_m = pd.DataFrame(df.corr().values) if corr is None else pd.DataFrame(corr.values)
        self.cov_m = pd.DataFrame(df.cov().values) if cov is None else pd.DataFrame(cov.values)
        self.col_names = self.df.columns if corr is None else corr.columns
        self.d_m = np.sqrt((1-corr_m)/2).values


    def _seriation(self, Z, N, cur_index):
        """Returns the order implied by a hierarchical tree (dendrogram).

           :param Z: A hierarchical tree (dendrogram).
           :param N: The number of points given to the clustering process.
           :param cur_index: The position in the tree for the recursive traversal.

           :return: The order implied by the hierarchical tree Z.
        """
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index - N, 0])
            right = int(Z[cur_index - N, 1])
            return (self._seriation(Z, N, left) + self._seriation(Z, N, right))

    def _compute_serial_mm(self, method="ward"):
        from scipy.linalg import block_diag
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        import matplotlib.pyplot as plt
        N = len(self.d_m)
        flat_d_m = squareform(self.d_m, checks=False)
        res_linkage = linkage(flat_d_m, method=method)
        res_order = self._seriation(res_linkage, N, N+N-2)
        s_d = np.zeros((N,N))
        a, b = np.triu_indices(N, k=1)
        s_d[a, b] = self.d_m[[res_order[i] for i in a], [res_order[j] for j in b]]
        s_d[b, a] = s_d[a, b]

        return s_d, res_order, res_linkage

    def _compute_HRP_weights(self, covariances, res_order):
        weights = pd.Series(1, index=res_order)
        clustered_alphas = [res_order]

        while len(clustered_alphas) > 0:
            clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2),
                                                   (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]
            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]
                left_subcovar = covariances[left_cluster].loc[left_cluster]
                inv_diag = 1 / np.diag(left_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

                right_subcovar = covariances[right_cluster].loc[right_cluster]
                inv_diag = 1 / np.diag(right_subcovar.values)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

                alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)

                weights[left_cluster] *= alloc_factor
                weights[right_cluster] *= 1 - alloc_factor

        return weights.sort_index().values

    def fit(self):
        s_d, res_order, res_linkage =self._compute_serial_mm()
        # zip(self._compute_HRP_weights(self.cov_m, res_order).sort_index(), self.col_names)
        # return self._compute_HRP_weights(self.cov_m, res_order).sort_index().values
        return pd.Series(self._compute_HRP_weights(self.cov_m, res_order), index=self.col_names)
        # return list(zip(self._compute_HRP_weights(self.cov_m, res_order).sort_index(), self.col_names))

class portfolio_allocation:

    def __init__(self, df=None, cov=None):

        self.cov_m = df.cov() if cov is None else cov

    def compute_MV_weights(self):
        inv_covar = np.linalg.inv(self.cov_m)
        u = np.ones(len(self.cov_m))

        return np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))

    def compute_RP_weights(self):
        weights = (1 / np.diag(self.cov_m))

        return weights / sum(weights)

    def compute_unif_weights(self):

        return [1 / len(self.cov_m) for i in range(len(self.cov_m))]


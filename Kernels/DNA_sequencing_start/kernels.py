import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

class SumKernel:
    def __init__(self, kernels, weights=None):
        """ Initialize with a list of kernels and optional weights """
        self.kernels = kernels
        self.weights = weights if weights else [1.0 for _ in kernels]

    def similarity(self, x, y):
        """ Compute the similarity between two strings using weighted sum of kernels """
        similarity_score = self.kernels[0].similarity(x, y) * self.weights[0]
        for i, kernel in enumerate(self.kernels[1:], 1):
            similarity_score += kernel.similarity(x, y) * self.weights[i]
        return similarity_score

    def gram(self, X1, X2=None):
        """ Compute the sum of the Gram matrices of all kernels """
        gram_matrix = self.kernels[0].gram(X1, X2) * self.weights[0]
        for i, kernel in tqdm(enumerate(self.kernels[1:], 1)):
            gram_matrix += kernel.gram(X1, X2) * self.weights[i]
        return gram_matrix


class LinearKernel:
    def __init__(self):
        """ Linear kernel initialization """
        pass

    def similarity(self, x, y):
        """ Linear kernel: k(x, y) = <x, y> """
        return np.dot(x, y)


class GaussianKernel:
    def __init__(self, sigma, normalize=True):
        """ Gaussian kernel initialization with optional normalization """
        self.sigma = sigma
        self.normalize = normalize

    def similarity(self, x, y):
        """ Gaussian kernel: k(x, y) = exp(-||x - y||^2 / 2 * sigma^2) / normalization factor """
        norm = np.linalg.norm(x - y)
        if self.normalize:
            norm_fact = (np.sqrt(2 * np.pi) * self.sigma) ** len(x)
            return np.exp(-norm**2 / (2 * self.sigma**2)) / norm_fact
        else:
            return np.exp(-norm**2 / (2 * self.sigma**2))


class PolynomialKernel:
    def __init__(self, gamma=1, coef0=1, degree=3):
        """ Polynomial kernel initialization """
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

    def similarity(self, x, y):
        """ Polynomial kernel: k(x, y) = (gamma * <x, y> + coef0)^degree """
        return (self.gamma * np.dot(x, y) + self.coef0) ** self.degree


class SpectrumKernel:
    def __init__(self, k):
        """ Spectrum kernel initialization """
        self.k = k

    def similarity(self, x, y):
        """ Spectrum kernel: compares the substrings of length 'k' """
        substr_x, counts_x = np.unique([x[i:i + self.k] for i in range(len(x) - self.k + 1)], return_counts=True)
        return np.sum(np.char.count(y, substr_x) * counts_x)


class MismatchKernel:
    def __init__(self, k, m, neighbours, kmer_set, normalize=False):
        """ Mismatch kernel initialization with optional normalization """
        self.k = k
        self.m = m
        self.kmer_set = kmer_set
        self.neighbours = neighbours
        self.normalize = normalize

    def neighbour_embed_kmer(self, x):
        """ Embed kmer with neighbours for a string x """
        kmer_x = [x[j:j + self.k] for j in range(len(x) - self.k + 1)]
        x_emb = {}
        for kmer in kmer_x:
            neigh_kmer = self.neighbours[kmer]
            for neigh in neigh_kmer:
                idx_neigh = self.kmer_set[neigh]
                x_emb[idx_neigh] = x_emb.get(idx_neigh, 0) + 1
        return x_emb

    def neighbour_embed_data(self, X):
        """ Embed data with neighbours for all strings in X """
        return [self.neighbour_embed_kmer(x) for x in X]

    def to_sparse(self, X_emb):
        """ Convert embedded data to a sparse matrix """
        data, row, col = [], [], []
        for i, x in enumerate(X_emb):
            data.extend(x.values())
            row.extend(x.keys())
            col.extend([i] * len(x))
        return sparse.coo_matrix((data, (row, col)))

    def similarity(self, x, y):
        """ Mismatch kernel similarity computation """
        x_emb = self.neighbour_embed_kmer(x)
        y_emb = self.neighbour_embed_kmer(y)
        similarity_score = sum(x_emb.get(idx, 0) * y_emb.get(idx, 0) for idx in x_emb)
        if self.normalize:
            norm_x = np.sqrt(sum(val**2 for val in x_emb.values()))
            norm_y = np.sqrt(sum(val**2 for val in y_emb.values()))
            similarity_score /= (norm_x * norm_y)
        return similarity_score

    def gram(self, X1, X2=None):
        """ Compute the Gram matrix for a dataset of strings """
        X1_emb = self.neighbour_embed_data(X1)
        X1_sm = self.to_sparse(X1_emb)
        if X2 is None:
            X2 = X1
        X2_emb = self.neighbour_embed_data(X2)
        X2_sm = self.to_sparse(X2_emb)
        nadd_row = abs(X1_sm.shape[0] - X2_sm.shape[0])
        if X1_sm.shape[0] > X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row-1], [X2_sm.shape[1]-1])))
            X2_sm = sparse.vstack((X2_sm, add_row))
        elif X1_sm.shape[0] < X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row-1], [X1_sm.shape[1]-1])))
            X1_sm = sparse.vstack((X1_sm, add_row))
        G = (X1_sm.T * X2_sm).todense().astype('float')
        if self.normalize:
            G /= np.array(np.sqrt(X1_sm.power(2).sum(0)))[0, :, None]
            G /= np.array(np.sqrt(X2_sm.power(2).sum(0)))[0, None, :]
        return G

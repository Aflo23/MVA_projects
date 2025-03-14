##### IMPORTS #####

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from kernels import LinearKernel, GaussianKernel, PolynomialKernel, SpectrumKernel, MismatchKernel, SumKernel
from utils import compute_or_load_neighbors
from classifiers import SVM

##### CONFIGURATION #####

# Define the chosen kernel and SVM parameters
kernel_type = 'sum'  # Options: 'sum', 'linear', 'rbf', 'poly', 'spectrum', 'mismatch'
C_param = 10.0       # Regularization parameter for SVM
gamma_value = 16.0   # Parameter gamma for RBF and Polynomial kernels
coef0_value = 1.0    # Parameter coef0 for Polynomial kernel
degree_value = 3     # Degree for Polynomial kernel
subseq_length = 12   # Subsequence length for spectrum and mismatch kernels
mismatch_penalty = 2 # Mismatch penalty for mismatch kernel
k_values = [5, 8, 10, 12, 13, 15] # List of subsequence lengths for sum of mismatch kernels
m_values = [1, 1, 1, 2, 2, 3]    # Mismatch penalties corresponding to k_values
kernel_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Weights for sum of mismatch kernels

# strings or not 
kernel_on_matrices = (kernel_type=='linear' or kernel_type=='rbf' or kernel_type=='poly')

# Debug print for the configuration
print(f"Kernel selected: {kernel_type}")
print(f"SVM C parameter: {C_param}")
if kernel_type in ['rbf', 'poly']:
    print(f"Gamma: {gamma_value}")
if kernel_type == 'poly':
    print(f"Coef0: {coef0_value}, Degree: {degree_value}")
if kernel_type == 'spectrum':
    print(f"K parameter: {subseq_length}")
if kernel_type == 'sum':
    print(f"List of k values: {k_values}")
    print(f"List of m values: {m_values}")
    print(f"Weights for sum of kernels: {kernel_weights}")
print()

##### DATA LOADING #####

# Loading training data (shapes: 2000,1 for strings and 2000,100 for feature matrices)
X0_train = pd.read_csv("data/Xtr0.csv", sep=",", index_col=0).values
X1_train = pd.read_csv("data/Xtr1.csv", sep=",", index_col=0).values
X2_train = pd.read_csv("data/Xtr2.csv", sep=",", index_col=0).values

X0_mat100_train = pd.read_csv("data/Xtr0_mat100.csv", sep=" ", header=None).values
X1_mat100_train = pd.read_csv("data/Xtr1_mat100.csv", sep=" ", header=None).values
X2_mat100_train = pd.read_csv("data/Xtr2_mat100.csv", sep=" ", header=None).values

X0_test = pd.read_csv("data/Xte0.csv", sep=",", index_col=0).values
X1_test = pd.read_csv("data/Xte1.csv", sep=",", index_col=0).values
X2_test = pd.read_csv("data/Xte2.csv", sep=",", index_col=0).values

X0_mat100_test = pd.read_csv("data/Xte0_mat100.csv", sep=" ", header=None).values
X1_mat100_test = pd.read_csv("data/Xte1_mat100.csv", sep=" ", header=None).values
X2_mat100_test = pd.read_csv("data/Xte2_mat100.csv", sep=" ", header=None).values

# Loading training labels
Y0_train = pd.read_csv("data/Ytr0.csv", sep=",", index_col=0).values
Y1_train = pd.read_csv("data/Ytr1.csv", sep=",", index_col=0).values
Y2_train = pd.read_csv("data/Ytr2.csv", sep=",", index_col=0).values

##### DATA PREPROCESSING #####

# Rescaling the labels (0 -> -1, 1 remains 1)
Y0_train = np.where(Y0_train == 0, -1, 1)
Y1_train = np.where(Y1_train == 0, -1, 1)
Y2_train = np.where(Y2_train == 0, -1, 1)

# Handling mismatch kernel computation if selected
if kernel_type == 'mismatch':
    neighbours_0, kmer_set_0 = compute_or_load_neighbors(0, subseq_length, mismatch_penalty)
    neighbours_1, kmer_set_1 = compute_or_load_neighbors(1, subseq_length, mismatch_penalty)
    neighbours_2, kmer_set_2 = compute_or_load_neighbors(2, subseq_length, mismatch_penalty)

# Verifying sum kernel parameters
if kernel_type == 'sum':
    assert(len(k_values) == len(m_values))
    assert(len(kernel_weights) == len(m_values))

# Shuffle data for better generalization
shuffle_data = True  # Set to False if you don't want shuffling

if shuffle_data:
    def shuffle_and_assign(X, Y):
        indices = np.random.permutation(len(X))
        return X[indices], Y[indices]
    
    X0_train, Y0_train = shuffle_and_assign(X0_train, Y0_train)
    X0_mat100_train = X0_mat100_train[np.random.permutation(len(X0_mat100_train))]
    # Apply to other datasets similarly...

##### APPLYING SVM #####

# Function to initialize and fit the SVM
def apply_svm(kernel_type, X_train, Y_train, X_test, kernel_params=None):
    if kernel_type == 'linear':
        svm = SVM(kernel=LinearKernel(), C=C_param)
    elif kernel_type == 'rbf':
        svm = SVM(kernel=GaussianKernel(sigma=np.sqrt(0.5 / gamma_value), normalize=False), C=C_param)
    elif kernel_type == 'poly':
        svm = SVM(kernel=PolynomialKernel(gamma=gamma_value, coef0=coef0_value, degree=degree_value), C=C_param)
    elif kernel_type == 'spectrum':
        svm = SVM(kernel=SpectrumKernel(k=subseq_length), C=C_param)
    elif kernel_type == 'mismatch':
        svm = SVM(kernel=MismatchKernel(k=subseq_length, m=mismatch_penalty, neighbours=kernel_params[0], kmer_set=kernel_params[1], normalize=True), C=C_param)
    elif kernel_type == 'sum':
        kernels = [MismatchKernel(k=k, m=m, neighbours=kernel_params[i][0], kmer_set=kernel_params[i][1], normalize=True) for i, (k, m) in enumerate(zip(k_values, m_values))]
        svm = SVM(kernel=SumKernel(kernels=kernels, weights=kernel_weights), C=C_param)

    svm.fit(X_train, Y_train)
    return svm.predict_classes(X_test)

# Running SVM on each dataset
if kernel_on_matrices: 

    pred_0 = apply_svm(kernel_type, X0_mat100_train, Y0_train, X0_test, kernel_params=[(neighbours_0, kmer_set_0) if kernel_type in ['mismatch', 'sum'] else None])
    pred_1 = apply_svm(kernel_type, X1_mat100_train, Y1_train, X1_test, kernel_params=[(neighbours_1, kmer_set_1) if kernel_type in ['mismatch', 'sum'] else None])
    pred_2 = apply_svm(kernel_type, X2_mat100_train, Y2_train, X2_test, kernel_params=[(neighbours_2, kmer_set_2) if kernel_type in ['mismatch', 'sum'] else None])

else: 
    pred_0 = apply_svm(kernel_type, X0_train, Y0_train, X0_test, kernel_params=[(neighbours_0, kmer_set_0) if kernel_type in ['mismatch', 'sum'] else None])
    pred_1 = apply_svm(kernel_type, X1_train, Y1_train, X1_test, kernel_params=[(neighbours_1, kmer_set_1) if kernel_type in ['mismatch', 'sum'] else None])
    pred_2 = apply_svm(kernel_type, X2_train, Y2_train, X2_test, kernel_params=[(neighbours_2, kmer_set_2) if kernel_type in ['mismatch', 'sum'] else None])



##### CREATE SUBMISSION FILE #####

# Concatenating predictions for all datasets
predictions = np.concatenate([pred_0, pred_1, pred_2])
predictions = np.where(predictions == -1, 0, 1)  # Convert -1 to 0 for final submission format
submission_df = pd.DataFrame({'Bound': predictions})
submission_df.index.name = 'Id'
submission_df.to_csv('Yte.csv', sep=',', header=True)

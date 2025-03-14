import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def generate_kmer_set(X, k, kmer_set={}):
    """
    Generates a set of all unique kmers in the dataset.
    """
    seq_length = len(X[0])
    idx = len(kmer_set)
    for sequence in X:
        kmers_in_sequence = [sequence[i:i + k] for i in range(seq_length - k + 1)]
        for kmer in kmers_in_sequence:
            if kmer not in kmer_set:
                kmer_set[kmer] = idx
                idx += 1
    return kmer_set


def find_neighbours(kmer, m, recursion_depth=0):
    """
    Returns a list of kmer neighbours with up to m mismatches.
    """
    if m == 0:
        return [kmer]

    nucleotides = ['G', 'T', 'A', 'C']
    k = len(kmer)
    neighbours = find_neighbours(kmer, m - 1, recursion_depth + 1)

    for neighbour in neighbours:
        for i in range(recursion_depth, k - m + 1):
            for nucleotide in nucleotides:
                neighbours.append(neighbour[:i] + nucleotide + neighbour[i + 1:])
    return list(set(neighbours))


def retrieve_neighbours(kmer_set, m):
    """
    Retrieve the neighbours for each kmer in the kmer_set.
    """
    kmers_list = list(kmer_set.keys())
    neighbours = {kmer: [] for kmer in kmers_list}

    for kmer in tqdm(kmers_list):
        kmer_neighbours = find_neighbours(kmer, m)
        for neighbour in kmer_neighbours:
            if neighbour in kmer_set:
                neighbours[kmer].append(neighbour)
    
    return neighbours


def load_precomputed_neighbors(dataset, k, m):
    """
    Load precomputed neighbours from a file.
    """
    file_name = f'neighbours_{dataset}_{k}_{m}.p'
    neighbours, kmer_set = pickle.load(open(f'saved_neighbors/{file_name}', 'rb'))
    print('Neighbors loaded successfully!')
    return neighbours, kmer_set


def compute_or_load_neighbors(dataset, k, m):
    """
    Either load precomputed neighbours or compute them if not found.
    """
    try:
        neighbours, kmer_set = load_precomputed_neighbors(dataset, k, m)
    except FileNotFoundError:
        print('No precomputed file found, creating new kmers and neighbours...')
        
        file_name = f'neighbours_{dataset}_{k}_{m}.p'
        if dataset == 0:
            X_train = pd.read_csv("data/Xtr0.csv", sep=",", index_col=0).values
            X_test = pd.read_csv("data/Xte0.csv", sep=",", index_col=0).values
            kmer_set = generate_kmer_set(X_train[:, 0], k)
            kmer_set = generate_kmer_set(X_test[:, 0], k, kmer_set)
            neighbours = retrieve_neighbours(kmer_set, m)
            pickle.dump([neighbours, kmer_set], open(f'saved_neighbors/{file_name}', 'wb'))
        
        elif dataset == 1:
            X_train = pd.read_csv("data/Xtr1.csv", sep=",", index_col=0).values
            X_test = pd.read_csv("data/Xte1.csv", sep=",", index_col=0).values
            kmer_set = generate_kmer_set(X_train[:, 0], k)
            kmer_set = generate_kmer_set(X_test[:, 0], k, kmer_set)
            neighbours = retrieve_neighbours(kmer_set, m)
            pickle.dump([neighbours, kmer_set], open(f'saved_neighbors/{file_name}', 'wb'))
        
        elif dataset == 2:
            X_train = pd.read_csv("data/Xtr2.csv", sep=",", index_col=0).values
            X_test = pd.read_csv("data/Xte2.csv", sep=",", index_col=0).values
            kmer_set = generate_kmer_set(X_train[:, 0], k)
            kmer_set = generate_kmer_set(X_test[:, 0], k, kmer_set)
            neighbours = retrieve_neighbours(kmer_set, m)
            pickle.dump([neighbours, kmer_set], open(f'saved_neighbors/{file_name}', 'wb'))

    return neighbours, kmer_set

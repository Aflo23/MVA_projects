"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


'''
let's be sure we use int in the generation of integers we create instead of floats 
'''

'''
build a 1st version now 
and then a second one 

optimize version of the 
--> probably can vectorize a lot of operations
'''


def create_train_dataset():
    n_train = 100000
    max_train_card = 10
    X_train = []
    y_train = []

    ############## Task 1
    
    ##################
    # your code here #
    for i in range(n_train):
        choice_number = np.random.randint(1,max_train_card+1) # choosing the number ofnon zero element we want 
        padded_array = np.concatenate((np.zeros(max_train_card-choice_number, dtype = int), np.random.randint(1,max_train_card, choice_number)))
        X_train.append(padded_array)
        y_train.append(int(np.sum(padded_array)))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    ##################

    return X_train, y_train





# def create_test_dataset():

#     ############## Task 2
    
#     ##################
#     # your code here #
#     n_test = int(100000*0.20)    
#     max_test_card = 10 

#     X_test, y_test = [], []
#     for i in range(n_test):
#         choice_number = np.random.randint(0,max_test_card) # choosing the number ofnon zero element we want 
#         padded_array = np.concatenate((np.zeros(max_test_card-choice_number, dtype = int), np.random.randint(1,max_test_card+1, choice_number)))
#         X_test.append(padded_array)
#         y_test.append(int(np.sum(padded_array)))

#     X_test = np.array(X_test)
#     y_test = np.array(y_test)
#     ##################

#     return X_test, y_test




def create_test_dataset():
    n_test = 200000  # Total number of samples
    max_test_card = 100  # Maximum number of digits in a multiset
    
    X_test, y_test = [], []
    
    for i in range(n_test):
        # Determine the number of digits for this batch (5, 10, 15, ..., 100)
        choice_number = 5 * ((i // 10000) + 1)  # 5, 10, 15, ..., 100 digits for each 10,000 samples
        
        # Generate the multiset: pad with zeros and add random digits
        padded_array = np.concatenate((
            np.zeros(max_test_card - choice_number, dtype=int),  # Padding with zeros
            np.random.randint(1, max_test_card + 1, choice_number)  # Random digits from 1 to 10
        ))
        
        X_test.append(padded_array)  # Add the sample to X_test
        y_test.append(int(np.sum(padded_array)))  # Sum of the elements in the sample for y_test
    
    X_test = np.array(X_test)  # Convert list to NumPy array
    y_test = np.array(y_test)  # Convert list to NumPy array
    
    return X_test, y_test

def create_test_dataset():
    n_test = 200000
    max_test_card = 100
    n_samples_per_card = 10000

    X_test = [np.zeros((n_samples_per_card, max_test_card), dtype=int) for _ in range(10)]
    y_test = [np.zeros(n_samples_per_card, dtype=int) for _ in range(10)]

    for i in range(5, max_test_card + 1, 5):
        for j in range(n_samples_per_card):
            X_test[i // 5 - 1][j, -i:] = np.random.randint(1, 11, i)
            y_test[i // 5 - 1][j] = np.sum(X_test[i // 5 - 1][j])


    return X_test, y_test

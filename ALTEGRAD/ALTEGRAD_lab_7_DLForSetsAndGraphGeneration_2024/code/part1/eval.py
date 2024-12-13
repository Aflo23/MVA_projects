"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
print(f'shape examples debuggng: {X_test[0].shape}, {type(X_test[0])}')
cards = [X_test[i].shape[0] for i in range(len(X_test))] # shape[0] before shape[1]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('/Users/antoine/Downloads/ALTEGRAD_lab_7_DLForSetsAndGraphGeneration_2024/code/model_deepsets.pth.tar') #     #code/model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('/Users/antoine/Downloads/ALTEGRAD_lab_7_DLForSetsAndGraphGeneration_2024/code/model_lstm.pth.tar')#code/model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

# for i in range(len(cards)):
#     y_pred_deepsets = list()
#     y_pred_lstm = list()
#     for j in range(0, n_samples_per_card, batch_size):
        
#         ############## Task 6
    
#         ##################
#         # your code here #

#         # Select the batch of data from X_test and the corresponding labels from y_test
#         x_batch = torch.tensor(X_test[i][j:j + batch_size]).to(device)  # Inputs for the batch
#         y_batch = torch.tensor([y_test[i]]*batch_size).to(device)  # [j:j + batch_size]Corresponding labels for the batch
        
#         # DeepSets model prediction
#         with torch.no_grad():  # Disable gradient tracking during evaluation
#             y_pred_deepsets_batch = deepsets(x_batch)  # Generate predictions
#             y_pred_deepsets.append(y_pred_deepsets_batch)  # Append predictions to the list
        
#         # LSTM model prediction
#         with torch.no_grad():  # Disable gradient tracking during evaluation
#             y_pred_lstm_batch = lstm(x_batch)  # Generate predictions
#             y_pred_lstm.append(y_pred_lstm_batch)  # Append predictions to the list
        
#         ##################
        
#     y_pred_deepsets = torch.cat(y_pred_deepsets)
#     y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
#     acc_deepsets = accuracy_score(y_test[i].cpu().numpy(), np.round(y_pred_deepsets)) #your code here
#     mae_deepsets = mean_absolute_error(y_test[i].cpu().numpy(), y_pred_deepsets) #your code here
#     results['deepsets']['acc'].append(acc_deepsets)
#     results['deepsets']['mae'].append(mae_deepsets)
    
#     y_pred_lstm = torch.cat(y_pred_lstm)
#     y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
#     acc_lstm = accuracy_score(y_test[i].cpu().numpy(), np.round(y_pred_lstm)) #your code here
#     mae_lstm =  mean_absolute_error(y_test[i].cpu().numpy(), y_pred_lstm) #your code here
#     results['lstm']['acc'].append(acc_lstm)
#     results['lstm']['mae'].append(mae_lstm)

itterable = tqdm(range(len(cards)))
for i in itterable:
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        # your code here #
        ##################
        x_batch = torch.tensor(X_test[i][j:j+batch_size]).to(device)

        output_deepset = deepsets(x_batch)
        y_pred_deepsets.append(output_deepset)

        output_lstm = lstm(x_batch)
        y_pred_lstm.append(output_lstm)
    
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
    acc_deepsets = accuracy_score(y_test[i], np.round(y_pred_deepsets)) #your code here
    mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets) #your code here
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(y_test[i], np.round(y_pred_lstm)) #your code here
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm) #your code here
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)
    
    itterable.set_description(f"Card: {cards[i]} - DeepSets: acc: {acc_deepsets:.4f}, mae: {mae_deepsets:.4f} - LSTM: acc: {acc_lstm:.4f}, mae: {mae_lstm:.4f}")





# for i in range(len(cards)):
#     y_pred_deepsets = list()
#     y_pred_lstm = list()
    
#     for j in range(0, n_samples_per_card, batch_size): #-batch_size peut etre ?
#         ############## Task 6

#         ##################
#         # Select the batch of data from X_test and the corresponding labels from y_test
#         print(f'X_test to sort : {X_test[i]}')
#         x_batch = torch.tensor(X_test[i][j:min(j+batch_size, n_samples_per_card), :]).to(device, dtype=torch.int64) # Inputs for the batch
#         #x_batch = torch.tensor(X_test[i][j:j + batch_size]).to(device)  # Inputs for the batch
#         print(f'X_test to sort 2 : {X_test[i][j:j + batch_size]}')
#         print(f'X-test shape {X_test[i].shape}')
#         # Correctly batch the corresponding labels from y_test
#         #y_batch = torch.tensor(y_test[i:i + batch_size]).to(device)  # Corresponding labels for the batch
#         print(f'deepset batch: {x_batch} shape {x_batch.shape}')
#         # DeepSets model prediction
#         with torch.no_grad():  # Disable gradient tracking during evaluation
#             y_pred_deepsets_batch = deepsets(x_batch)  # Generate predictions

#             if y_pred_deepsets_batch.dim() == 0:  # scalar
#                 y_pred_deepsets_batch = y_pred_deepsets_batch.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1)

#             y_pred_deepsets.append(y_pred_deepsets_batch)  # Append predictions to the list
        
#         # LSTM model prediction
#         with torch.no_grad():  # Disable gradient tracking during evaluation
#             y_pred_lstm_batch = lstm(x_batch)  # Generate predictions
#             y_pred_lstm.append(y_pred_lstm_batch)  # Append predictions to the list
        
#         ##################
    
#     # Convert predictions to a single tensor

#     y_pred_deepsets = torch.cat(y_pred_deepsets)
#     y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
#     #acc_deepsets = accuracy_score(y_test[i].cpu().numpy(), np.round(y_pred_deepsets))
#     # mae_deepsets = mean_absolute_error(y_test[i].cpu().numpy(), y_pred_deepsets)


#     acc_deepsets = accuracy_score(y_test[i], np.round(y_pred_deepsets))
#     mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)
#     results['deepsets']['acc'].append(acc_deepsets)
#     results['deepsets']['mae'].append(mae_deepsets)
    
#     y_pred_lstm = torch.cat(y_pred_lstm)
#     y_pred_lstm = torch.round(y_pred_lstm)
#     y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
#     # acc_lstm = accuracy_score(y_test[i].cpu().numpy(), np.round(y_pred_lstm))
#     # mae_lstm = mean_absolute_error(y_test[i].cpu().numpy(), y_pred_lstm)

#     acc_lstm = accuracy_score(np.array([y_test[i]]), np.round(y_pred_lstm))
#     mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
#     results['lstm']['acc'].append(acc_lstm)
#     results['lstm']['mae'].append(mae_lstm)



############## Task 7
    
##################
# your code here #

# x axis 
cardinalities = [5 * (i + 1) for i in range(len(results['deepsets']['acc']))]

plt.figure(figsize=(16,10))
plt.title("accuracy scores lstm & deepsets")
plt.plot(cardinalities, results['lstm']['acc'], marker = 'x', color = 'g', label = "lstm")
plt.plot(cardinalities, results['deepsets']['acc'], marker = 'x', color = 'r', label = 'deepsets')

plt.xlabel("Maximum Cardinality of Input Sets")
plt.ylabel("Accuracy Score ")

plt.grid(True)
plt.show()

##################

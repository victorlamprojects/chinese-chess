# Global Config
from common.GlobalConfig import *
import pickle
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
import tqdm

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, seq, y):
        assert len(seq) == len(y)
        self.seq = seq
        self.y = y
    def __getitem__(self, idx):
        return self.seq[idx], self.y[idx]
    def __len__(self):
        return len(self.seq)

def save_model(model, name):
    return torch.save(model, f"./best_model/{name}.pth")
def load_model(name):
    return torch.load(f"./best_model/{name}.pth")
def save_model_info(d, name):
    with open(f'./best_model/{name}.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
def get_clean_params(params):
    nP = params.copy()
    nP['hidden_size'] = int(nP['hidden_size'])
    nP['num_filters'] = int(nP['num_filters'])
    nP['num_mlp_layers'] = int(nP['num_mlp_layers'])
    nP['num_cnn_layers'] = int(nP['num_cnn_layers'])
    nP['pool_size'] = int(nP['pool_size'])
    nP['num_of_epochs'] = int(nP['num_of_epochs'])
    nP['kernel_size'] = kernel_sizes_map[int(nP['kernel_index'])-1]
    nP.pop('kernel_index', None)
    return nP
def check_valid_cnn_output_size(init_size, n_layers, kernel_sizes, pool_size,stride=1):
    for i in range(len(kernel_sizes)):
        o_size = init_size
        o_size = np.floor((o_size - kernel_sizes[i])/stride) + 1
        if o_size <= 0:
            return False
        for j in range(n_layers-1):
            o_size = np.floor((o_size - pool_size)/pool_size) + 1
            if o_size <= 0:
                return False
            o_size = np.floor((o_size - kernel_sizes[i])/stride) + 1
            if o_size <= 0:
                return False
    return True
def get_loss_function(x):
    if x == 'BCE':
        def loss(pred, actual):
            loss_func = torch.nn.BCEWithLogitsLoss()
            correct_count = ((pred>0.5).float() == actual).sum().item()
            return (loss_func(pred, actual), correct_count)
        return loss
    else:
        def loss(pred, actual):
            loss_func = torch.nn.CrossEntropyLoss()
            correct_count = (pred.argmax(1) == actual).sum().item()
            return (loss_func(pred, actual), correct_count)
        return loss
def predict(model, test):
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    y_true = []
    y_pred = []
    y_true_cat = []
    y_pred_cat = []
    with tqdm.tqdm(test) as t:
        for x, y in t:
            x_ = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            logits = model(x_)
            y_true = y_true + y.tolist()
            y_pred = y_pred + (logits>0.5).float().tolist()
    return y_true, y_pred
def print_classification_report(y_pred, y_true, cat_list, title='Classification Report:'):
    print(title)
    print(classification_report(y_true, y_pred, target_names=cat_list))
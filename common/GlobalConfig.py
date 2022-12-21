import itertools

random_seed = 42
split_ratio=0.2
batch_size = 16
num_epoch = 20

data_folder_path = f'./data/raw'
clean_folder_path = f'./data/clean_data'
preprocessed_folder_path = f'./data/preprocessed_data'

chess_types = ['r', 'n', 'b', 'a', 'k', 'c', 'p']

kernel_sizes_map = []
kernel_sizes = [1,2,3,4,5]
for i in range(1,len(kernel_sizes)+1):
    kernel_sizes_map += list(itertools.combinations(kernel_sizes,i))

param_grid = {
    'hidden_size': (16, 128),
    'dropout_rate': (0.1, 0.5), 
    'learning_rate': (0.001, 0.01), 
    'l2_reg': (0.0001, 0.001),
    'num_filters': (2, 16),
    'pool_size': (2, 16),
    'num_mlp_layers': (1, 5),
    'num_cnn_layers': (1, 5),
    'num_lstm_layers': (1, 5),
    'kernel_index': (1,31),
    'sigmoid': (0, 1),
    'num_of_epochs': (5, 15)
}
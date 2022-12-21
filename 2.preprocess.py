# Global Config
from common.GlobalConfig import *
from common.Util import *
import pandas as pd
import numpy as np
from random import shuffle
import os
if not os.path.exists(preprocessed_folder_path):
    os.makedirs(preprocessed_folder_path)

from sklearn.model_selection import train_test_split

# Helper functions
def load_data(file_name=''):
    if file_name != '':
        return pd.read_csv(f'{clean_folder_path}/{file_name}')
def get_games():
    files = []
    for directory in os.walk(clean_folder_path):
        files = files + directory[2]
    return files
def split_dataframe(df, chunk_size=10000):
    num_of_chunks = len(df)//chunk_size + 1
    dfs = []
    for i in range(num_of_chunks):
        dfs.append(df.iloc[i*chunk_size:(i+1)*chunk_size])
    return dfs
    
def convert_game_to_dict(game, turn):
    return {
        'player': turn,
        'result': game.columns[0],
        'gameboard': game.to_numpy()
    }
def convert_game_to_dicts(game):
    game_dicts = []
    gameboards = split_dataframe(game, 10)
    turn = 'r'
    for gb in gameboards:
        if gb.shape != (10, 9):
            continue
        game_dicts.append(convert_game_to_dict(gb, turn))
        if turn == 'r':
            turn = 'b'
        else:
            turn = 'r'
    return game_dicts
def encode_chess_type(chess):
    return chess_types.index(chess) + 1
def encode_gameboard(game):
    board = np.asarray(game['gameboard'])
    win = 1 if game['player'] == game['result'] else 0
    player = game['player']
    opponent = 'b' if player == 'r' else 'b'
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c][0] == player:
                board[r][c] = encode_chess_type(board[r][c][1])
            elif board[r][c][0] == opponent:
                board[r][c] = -1 * encode_chess_type(board[r][c][1])
            else:
                board[r][c] = 0
    return board, win
def preprocess():
    # Get a list of game ids
    game_files= get_games()
    data = []
    for game_file in game_files:
        game_id = game_file.split('.')[0]
        game = load_data(game_file)
        results = convert_game_to_dicts(game)
        for i in range(len(results)):
            d = encode_gameboard(results[i])
            data.append(d)
    return train_test_split(data, test_size=split_ratio)

# Preprocess
train, test = preprocess()

# Shuffle
shuffle(train)
shuffle(test)

# Split input and output
train_X = np.asarray([t[0] for t in train], dtype=np.float32)
train_Y = np.asarray([t[1] for t in train], dtype=np.float32).reshape(-1,1)
test_X = np.asarray([t[0] for t in test], dtype=np.float32)
test_Y = np.asarray([t[1] for t in test], dtype=np.float32).reshape(-1,1)

## Save encoded data
np.save(f'{preprocessed_folder_path}/train_X.npy', train_X)
np.save(f'{preprocessed_folder_path}/test_X.npy', test_X)
np.save(f'{preprocessed_folder_path}/train_Y.npy', train_Y)
np.save(f'{preprocessed_folder_path}/test_Y.npy', test_Y)
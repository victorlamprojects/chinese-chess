# Global Config
from common.GlobalConfig import *
import pandas as pd
import os
if not os.path.exists(clean_folder_path):
    os.makedirs(clean_folder_path)
# Helper functions
def load_data(file_name=''):
    if file_name != '':
        return pd.read_csv(f'{data_folder_path}/{file_name}')
def get_games():
    files = []
    for directory in os.walk(data_folder_path):
        files = files + directory[2]
    return files
def remove_first_step(game):
    removed = game[10:]
    return removed
# Get a list of game ids
gameIds = get_games()
# Remove first step as initial game board is neutral
for file in gameIds:
    board = load_data(file)
    board.columns = [board.columns[0],"","","","","","","",""]
    cleaned_game = remove_first_step(board)
    cleaned_game.to_csv(f"{clean_folder_path}/{file}", index=False)
import math
from copy import deepcopy
import concurrent.futures
from common.GlobalConfig import chess_types
from common.Util import load_model
import torch


# King and Advisor can go to 3 only
# Bishop can go to >= 2
# Others can go >= 1
VALIDS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 3, 3, 3, 1, 1, 1, 0, 0],
    [0, 0, 2, 1, 1, 3, 3, 3, 1, 1, 2, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 1, 0, 0],
    [0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 2, 1, 1, 3, 3, 3, 1, 1, 2, 0, 0],
    [0, 0, 1, 1, 1, 3, 3, 3, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 2, 3, 3, 3, 2, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
N_MOVE = [[-2, -1], [-2, 1], [2, -1], [2, 1], [-1, -2], [1, -2], [-1, 2], [1, 2]]
B_MOVE = [[-2, -2], [2, -2], [-2, 2], [2, 2]]
A_MOVE = [[1, -1], [-1, 1], [-1, -1], [1, 1]]
K_MOVE = [[-1, 0], [1, 0], [0, -1], [0, 1]]
MOVE_MAP = {"k": K_MOVE, "a": A_MOVE, "b": B_MOVE, "n": N_MOVE}


def get_opponent(current_player) -> str:
    return "r" if current_player == "b" else "b"


def get_king_position(current_player, board) -> tuple:
    for i in range(10):
        for j in range(3, 6):
            if board[i][j] == current_player + "k":
                return (i, j)


def calculate_king_advisory_possible_moves(i, j, board) -> list:
    moves = []
    current_player = board[i][j][0]
    for dr, dc in MOVE_MAP[board[i][j][1]]:
        r = dr + i
        c = dc + j
        if VALIDS[r + 2][c + 2] == 3 and board[r][c][0] != current_player:
            moves.append((r, c))
    return moves


def calculate_bishop_possible_moves(i, j, board) -> list:
    moves = []
    current_player = board[i][j][0]
    for dr, dc in MOVE_MAP[board[i][j][1]]:
        r = dr + i
        c = dc + j
        if (
            VALIDS[r + 2][c + 2] >= 2
            and board[r][c][0] != current_player
            and board[i + int(math.fmod((1 + dr), 2))][j + int(math.fmod((1 + dc), 2))]
            == "--"
        ):
            moves.append((r, c))
    return moves


def calculate_cannon_possible_moves(i, j, board) -> list:
    def get_next_chess(r, c, dr, dc, board):
        r, c = r + dr, c + dc
        while VALIDS[r + 2][c + 2]:
            if board[r][c] != "--":
                return (r, c)
            r, c = r + dr, c + dc
        return None

    moves = []
    current_player = board[i][j][0]
    opponent = get_opponent(current_player)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in dirs:
        r, c = i + dr, j + dc
        while VALIDS[r + 2][c + 2]:
            if board[r][c] != "--":
                next_chess = get_next_chess(r, c, dr, dc, board)
                if next_chess and board[next_chess[0]][next_chess[1]][0] == opponent:
                    moves.append(next_chess)
                break
            moves.append((r, c))
            r, c = r + dr, c + dc
    return moves


def calculate_knight_possible_moves(i, j, board) -> list:
    moves = []
    current_player = board[i][j][0]
    for dr, dc in MOVE_MAP[board[i][j][1]]:
        r = dr + i
        c = dc + j
        if (
            VALIDS[r + 2][c + 2]
            and board[r][c][0] != current_player
            and board[i + int(math.fmod(1 + dr, 2))][j + int(math.fmod(1 + dc, 2))]
            == "--"
        ):
            moves.append((r, c))
    return moves


def calculate_rook_possible_moves(i, j, board) -> list:
    moves = []
    current_player = board[i][j][0]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in dirs:
        r, c = i + dr, j + dc
        while VALIDS[r + 2][c + 2] and board[r][c][0] != current_player:
            moves.append((r, c))
            if board[r][c] != "--":
                break
            r, c = r + dr, c + dc
    return moves


def calculate_pawn_possible_moves(i, j, board) -> list:
    moves = []
    current_player = board[i][j][0]
    move_dir = 1
    if get_king_position(current_player, board)[0] > 4:
        move_dir = -1
    # Checking forward movement
    if VALIDS[i + move_dir + 2][j + 2] and board[i + move_dir][j][0] != current_player:
        moves.append((i - 1, j))
    # Checking left-right movement
    if (move_dir == 1 and i >= 5) or (move_dir == -1 and i <= 4):
        if VALIDS[i + 2][j - 1 + 2] and board[i][j - 1][0] != current_player:
            moves.append((i, j - 1))
        if VALIDS[i + 2][j + 1 + 2] and board[i][j + 1][0] != current_player:
            moves.append((i, j + 1))
    return moves


def calculate_possible_moves(i, j, board) -> list:
    if board[i][j] == "--":
        return []
    if board[i][j][1] == "k" or board[i][j][1] == "a":
        return calculate_king_advisory_possible_moves(i, j, board)
    elif board[i][j][1] == "b":
        return calculate_bishop_possible_moves(i, j, board)
    elif board[i][j][1] == "n":
        return calculate_knight_possible_moves(i, j, board)
    elif board[i][j][1] == "r":
        return calculate_rook_possible_moves(i, j, board)
    elif board[i][j][1] == "c":
        return calculate_cannon_possible_moves(i, j, board)
    return calculate_pawn_possible_moves(i, j, board)


def is_being_checked(p, board) -> bool:
    king = get_king_position(p, board)
    opponent = get_opponent(p)
    for i in range(10):
        for j in range(9):
            if board[i][j][0] == opponent:
                if board[i][j][1] == "k":
                    if j == king[1]:
                        l = i + 1
                        u = king[0]
                        if l > u:
                            l = king[0] + 1
                            u = i
                        while l < u and board[l][j] == "--":
                            l += 1
                        if l == u:
                            return True
                        continue
                moves = calculate_possible_moves(i, j, board)
                for move in moves:
                    if king == move:
                        return True
    return False


def get_valid_moves(row, col, current_player, board) -> list:
    if board[row][col][0] != current_player:
        return []
    board_copy = deepcopy(board)
    valid_moves = []
    for x, y in calculate_possible_moves(row, col, board_copy):
        destChess = board_copy[x][y]
        board_copy[x][y], board_copy[row][col] = board_copy[row][col], "--"
        if not is_being_checked(current_player, board_copy):
            valid_moves.append((x, y))
        board_copy[x][y], board_copy[row][col] = destChess, board_copy[x][y]
    return valid_moves


def is_checkmated(current_player, board) -> bool:
    for i in range(10):
        for j in range(9):
            if board[i][j][0] == current_player:
                valid_moves = get_valid_moves(i, j, current_player, board)
                if valid_moves:
                    return False
    return True


def get_valid_moves_map(current_player, board):
    valid_moves_map = {}
    for i in range(10):
        for j in range(9):
            if board[i][j][0] == current_player:
                valid_moves_map[(i, j)] = get_valid_moves(i, j, current_player, board)
    return valid_moves_map


def calculate_score(current_player, board):
    model = load_model(f"best_cnn_model")
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model(torch.Tensor([encode_gameboard(current_player, board)]))


def calculate_score_for_x_y(i, j, x, y, current_player, board):
    board[x][y], board[i][j] = board[i][j], "--"
    score = calculate_score(current_player, board)
    # print(f"origin: {i} {j}, dest: {x}, {y}, score: {score}")
    return score


def get_best_move_and_score_for_i_j(i, j, valid_moves, current_player, board):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_move = {
            executor.submit(
                calculate_score_for_x_y, i, j, x, y, current_player, deepcopy(board)
            ): (x, y)
            for x, y in valid_moves
        }
        best_move, best_score = (-1, -1), -1
        for future in concurrent.futures.as_completed(future_to_move):
            move = future_to_move[future]
            score = future.result()
            if score > best_score:
                best_move, best_score = move, score
        return best_move, best_score


def get_best_move(current_player, board):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pos = {
            executor.submit(
                get_best_move_and_score_for_i_j,
                i,
                j,
                valid_moves,
                current_player,
                board,
            ): (i, j)
            for (i, j), valid_moves in get_valid_moves_map(
                current_player, board
            ).items()
        }
        best_move, best_score = [(-1, -1), (-1, -1)], -1
        for future in concurrent.futures.as_completed(future_to_pos):
            pos = future_to_pos[future]
            move, score = future.result()
            if score > best_score:
                best_move, best_score = [pos, move], score
    return best_move


def encode_gameboard(current_player, board):
    def encode_chess_type(chess):
        return chess_types.index(chess) + 1

    opponent = get_opponent(current_player)
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c][0] == current_player:
                board[r][c] = encode_chess_type(board[r][c][1])
            elif board[r][c][0] == opponent:
                board[r][c] = -1 * encode_chess_type(board[r][c][1])
            else:
                board[r][c] = 0
    return board

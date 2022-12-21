# King and Advisor can go to 3 only
# Bishop can go to >= 2
# Others can go >= 1
VALIDS = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Deviation of all chesses' move
H_MOVE =[[-2,-1], [-2,1], [2,-1],
         [2,1], [-1,-2], [1,-2],
         [-1,2], [1,2]]
B_MOVE =[[-2,-2], [2,-2], [-2,2], [2,2]]
A_MOVE =[[1,-1], [-1,1], [-1,-1], [1,1]];
K_MOVE =[[-1,0], [1,0], [0,-1], [0,1]];

def get_opponent(p):
    if p == "b":
        return "r"
    return "b"
def get_king(p, board):
    opponent = get_opponent(p)
    king = None
    for i in range(10):
        for j in range(3,6):
            if board[i][j] == p + "k":
                king = (i,j)
    return king
def get_next_opponent(x, y, dir_i, dir_j, p, board):
    x += dir_i
    y += dir_j
    while VALIDS[x+2][y+2] >= 1:
        if board[x][y] == get_opponent(p, board):
            return (x, y)
        x += dir_i
        y += dir_j
    return (-1,-1)
# Return a list of possible moves of a chess
def calculate_moves(i, j, board):
    if board[i][j] == "--":
        return []
    turn = board[i][j][0]
    opponent = get_opponent(turn)
    valid_moves = []
    m_map = {
        "k": K_MOVE,
        "a": A_MOVE,
        "b": B_MOVE,
        "h": H_MOVE
    }
    # Check King and Advisory(==3)
    if board[i][j][1] == "k" or board[i][j][1] == "a":
        for m in m_map[board[i][j][1]]:
            r = m[0] + i
            c = m[1] + j
            if VALIDS[r+2][c+2] == 3 and board[r][c][0] != turn:
                valid_moves.append((r,c))
    # Check Elephant(>=2)
    elif board[i][j][1] == "b":
        for m in m_map[board[i][j][1]]:
            r = m[0] + i
            c = m[1] + j
            if VALIDS[r+2][c+2] >= 2 and board[r][c][0] != turn and board[i+((1+m[0])%2)][j+((1+m[1])%2)] == "--":
                valid_moves.append((r,c))
    # Check Others(>=1)
    elif board[i][j][1] == "h":
        for m in m_map[board[i][j][1]]:
            r = m[0] + i
            c = m[1] + j
            if VALIDS[r+2][c+2] >= 1 and board[r][c][0] != turn and board[i+((1+m[0])%2)][j+((1+m[1])%2)]=="--":
                valid_moves.append((r,c))
    elif board[i][j][1] == "r":
        r = i
        c = j
        # Check vertical direction
        r = i - 1
        while VALIDS[r+2][c+2] >= 1 and board[r][c][0] != turn:
            valid_moves.append((r, c))
            # stop here
            if board[r][c] != "--":
                break
            r -= 1
        r = i + 1
        while VALIDS[r+2][c+2] and board[r][c][0] != turn:
            valid_moves.append((r, c))
            # stop here
            if board[r][c] != "--":
                break
            r += 1
        # Check horizontal direction
        r = i
        c = j - 1
        while VALIDS[r+2][c+2] and board[r][c][0] != turn:
            valid_moves.append((r,c))
            # stop here
            if board[r][c] != "--":
                break
            c -= 1
        c = j + 1
        while VALIDS[r+2][c+2] and board[r][c][0] != turn:
            valid_moves.append((r,c))
            # stop here
            if board[r][c] != "--":
                break
            c += 1
    elif board[i][j][1] == "c":
        # Check vertical direction
        r = i - 1
        c = j
        while VALIDS[r+2][c+2]:
            # stop here
            if board[r][c] != "--":
                temp = get_next_opponent(r, c, -1, 0, turn, board)
                if temp[0] != -1 and temp[1] != -1 and board[temp[0]][temp[1]][0] == opponent:
                    valid_moves.append(temp)
                break
            valid_moves.append((r,c))
            r -= 1
        r = i + 1
        while VALIDS[r+2][c+2]:
            # stop here
            if board[r][c] != "--":
                temp = get_next_opponent(r, c, 1, 0, turn, board)
                if temp[0] != -1 and temp[1] != -1 and board[temp[0]][temp[1]][0] == opponent:
                    valid_moves.append(temp)
                break
            valid_moves.append((r,c))
            r += 1
        # Check horizontal direction
        r = i
        c = j - 1
        while VALIDS[r+2][c+2]:
            # stop here
            if board[r][c] != "--":
                temp = get_next_opponent(r, c, 0, -1, p, board)
                if temp[0] != -1 and temp[1] != -1 and board[temp[0]][temp[1]][0] == opponent:
                    valid_moves.append(temp)
                break
            valid_moves.append((r,c))
            c -= 1
        c = j + 1
        while VALIDS[r+2][c+2]:
            # stop here
            if board[r][c] != "--":
                temp = get_next_opponent(r, c, 0, 1, p, board)
                if temp[0] != -1 and temp[1] != -1 and board[temp[0]][temp[1]][0] == opponent:
                    valid_moves.append(temp)
                break
            valid_moves.append((r,c))
            c += 1
    elif board[i][j][1] == "p":
        move_dir = 1
        if get_king(turn, board)[0] > 4:
            move_dir = -1
        # Checking forward movement
        if VALIDS[i+move_dir+2][j+2] and board[i+move_dir][j][0] != turn:
            valid_moves.append((i-1, j))
        # Checking left-right movement
        if (move_dir == 1 and i >= 5) or (move_dir == -1 and i <= 4):
            if VALIDS[i+2][j-1+2] and board[i][j-1][0] != turn:
                valid_moves.append((i, j-1))
            if VALIDS[i+2][j+1+2] and board[i][j+1][0] != turn:
                valid_moves.append((i, j+1))
    return valid_moves
# Check if the current player is in check in the chessboard
def is_checked(p, board):
    # Get King's position of current player
    king = get_king(p, board)
    # Check if opponent's chess can kill the king
    for i in range(10):
        for j in range(9):
            if board[i][j][0] == opponent:
                # Check special case
                if board[i][j][1] == "k":
                    if j == king[1]:
                        l = i + 1
                        u = king[0]
                        if l > u:
                            l = king[0] + 1
                            u = i
                        while l < u and board[i][j] == "--":
                            l += 1
                        if l == u:
                            return True
                        break
                moves = calculate_moves(i, j, board)
                for move in range(len(moves)):
                    if king == move:
                        return True
    return False

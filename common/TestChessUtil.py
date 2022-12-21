from ChessUtil import (
    get_king_advisory_possible_moves,
    get_bishop_possible_moves,
    get_knight_possible_moves,
    get_rook_possible_moves,
    get_cannon_possible_moves,
    get_pawn_possible_moves,
    calculate_moves,
    is_checked,
    get_valid_moves_from_pseudo_moves,
    is_checkmated,
    get_valid_moves,
    get_best_move
)
import unittest


class TestChessUtil(unittest.TestCase):
    def setUp(self):
        self.board = [
            ["br", "bn", "bb", "ba", "bk", "ba", "bb", "bn", "br"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "bc", "--", "--", "--", "--", "--", "bc", "--"],
            ["bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["rp", "--", "rp", "--", "rp", "--", "rp", "--", "rp"],
            ["--", "rc", "--", "--", "--", "--", "--", "rc", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["rr", "rn", "rb", "ra", "rk", "ra", "rb", "rn", "rr"],
        ]
        self.board2 = [
            ["br", "bn", "bb", "ba", "bk", "ba", "bb", "bn", "br"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "bc", "--", "--", "--", "--", "--", "bc", "--"],
            ["bp", "--", "bp", "--", "rp", "--", "bp", "--", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "rb", "--", "--", "--", "--", "--", "--"],
            ["rp", "--", "rp", "--", "--", "--", "rp", "--", "rp"],
            ["--", "rc", "--", "--", "rc", "--", "rn", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["rr", "rn", "--", "ra", "rk", "ra", "rb", "--", "rr"],
        ]
        self.board3 = [
            ["br", "bn", "bb", "--", "bk", "--", "bb", "bn", "br"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "rp", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["rr", "rn", "--", "--", "rk", "--", "--", "--", "rr"],
        ]
        self.board4 = [
            ["br", "bn", "bb", "--", "bk", "--", "bb", "bn", "br"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["rr", "rn", "--", "--", "rk", "--", "--", "--", "rr"],
        ]
        self.board5 = [
            ["--", "--", "--", "bk", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "rr", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "rk", "--", "--", "--", "--"],
        ]

    def test_get_king_possible_moves(self):
        i, j = 0, 4
        moves = get_king_advisory_possible_moves(i, j, self.board)
        expected_moves = [(1, 4)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_advisory_possible_moves(self):
        i, j = 0, 3
        moves = get_king_advisory_possible_moves(i, j, self.board)
        expected_moves = [(1, 4)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_bishop_possible_moves(self):
        i, j = 0, 2
        moves = get_bishop_possible_moves(i, j, self.board)
        expected_moves = [(2, 0), (2, 4)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_bishop_river_possible_moves(self):
        i, j = 5, 2
        moves = get_bishop_possible_moves(i, j, self.board2)
        expected_moves = [(7, 0)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_knight_possible_moves(self):
        i, j = 9, 1
        moves = get_knight_possible_moves(i, j, self.board)
        expected_moves = [(7, 0), (7, 2)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_knight_blocked_possible_moves(self):
        i, j = 7, 6
        moves = get_knight_possible_moves(i, j, self.board2)
        expected_moves = [(9, 7), (8, 8), (8, 4), (6, 4)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_rook_possible_moves(self):
        i, j = 9, 0
        moves = get_rook_possible_moves(i, j, self.board)
        expected_moves = [(8, 0), (7, 0)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_cannon_possible_moves(self):
        i, j = 7, 1
        moves = get_cannon_possible_moves(i, j, self.board)
        expected_moves = [
            (7, 0),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (0, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (8, 1),
        ]
        self.assertCountEqual(expected_moves, moves)

    def test_get_pawn_possible_moves(self):
        i, j = 6, 4
        moves = get_pawn_possible_moves(i, j, self.board)
        expected_moves = [(5, 4)]
        self.assertCountEqual(expected_moves, moves)

    def test_calculate_moves(self):
        i, j = 9, 0
        moves = calculate_moves(i, j, self.board)
        expected_moves = [(8, 0), (7, 0)]
        self.assertCountEqual(expected_moves, moves)

    def test_is_checked_true(self):
        self.assertTrue(is_checked(p="b", board=self.board2))

    def test_is_checked_false(self):
        self.assertFalse(is_checked(p="b", board=self.board3))
    
    def test_is_checked_special_case(self):
        self.assertTrue(is_checked(p="b", board=self.board4))

    def test_get_valid_moves_from_pseudo_moves_cant_move_case(self):
        i, j = 0, 0
        moves = get_valid_moves_from_pseudo_moves(i, j, calculate_moves(i, j, self.board2), "b", self.board2)
        expected_moves = []
        self.assertCountEqual(expected_moves, moves)
    
    def test_get_valid_moves_from_pseudo_moves_defend_king_case(self):
        i, j = 0, 2
        moves = get_valid_moves_from_pseudo_moves(i, j, calculate_moves(i, j, self.board2), "b", self.board2)
        expected_moves = [(2, 4)]
        self.assertCountEqual(expected_moves, moves)

    def test_is_checkmated_true(self):
        self.assertTrue(is_checkmated("b", self.board5))
    
    def test_is_checkmated_false(self):
        self.assertFalse(is_checkmated("b", self.board2))

    def test_get_valid_moves(self):
        i, j = 0, 2
        moves = get_valid_moves(i, j, "b", self.board2)
        expected_moves = [(2, 4)]
        self.assertCountEqual(expected_moves, moves)

    def test_get_best_move(self):
        expected_best_move = []
        best_move = get_best_move("b", self.board2)
        self.assertCountEqual(expected_best_move, best_move)

if __name__ == "__main__":
    unittest.main()

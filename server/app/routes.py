from app import app
from flask import request
from common.ChessUtil import get_best_move


@app.route("/best-move", methods=["POST"])
def best_move():
    data = request.json
    current_player, board = data["player"], data["board"]
    return {"best_move": get_best_move(current_player, board)}

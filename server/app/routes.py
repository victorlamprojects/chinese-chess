from app import app
from flask import request, Response
from flask_cors import CORS, cross_origin
from common.ChessUtil import get_best_move
import json


@app.route("/best-move", methods=["POST"])
@cross_origin(origins="*")
def best_move():
    data = request.json
    current_player, board = data["player"], data["board"]
    response = {"best_move": get_best_move(current_player, board)}
    return Response(json.dumps(response), mimetype="application/json", status=200)

// Variables needed
var player = "r";

// King and Advisor can go to 3 only
// Elephant can go to >= 2
// Others can go >= 1
const valids = [
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
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
];
const shift = 2;

// Deviation of all chesses' move
const n_move = [[-2, -1],
[-2, 1],
[2, -1],
[2, 1],
[-1, -2],
[1, -2],
[-1, 2],
[1, 2]];
const b_move = [[-2, -2],
[2, -2],
[-2, 2],
[2, 2]];
const a_move = [[1, -1],
[-1, 1],
[-1, -1],
[1, 1]];
const k_move = [[-1, 0],
[1, 0],
[0, -1],
[0, 1]];

// Scoresheet
const rs = [   // soldiers
	[9, 9, 9, 11, 13, 11, 9, 9, 9],
	[19, 20, 34, 42, 44, 42, 34, 20, 19],
	[19, 21, 30, 35, 37, 35, 30, 21, 19],
	[19, 20, 25, 27, 30, 27, 25, 20, 19],
	[14, 18, 20, 23, 26, 23, 20, 18, 14],
	[7, 0, 10, 0, 16, 0, 10, 0, 7],
	[7, 0, 7, 0, 15, 0, 7, 0, 7],
	[0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0]
]

// Duplicate Chessboard
var dup_chessboard = function (chessboard) {
	const cb = chessboard;
	var new_board = (new Array(10)).fill().map(function () {
		return new Array(9).fill(null);
	});
	for (let i = 0; i < 10; i++) {
		for (let j = 0; j < 9; j++) {
			new_board[i][j] = cb[i][j];
		}
	}
	return new_board;
}

// Return a list of possible moves(not yet check the checkmate)
var calculate_moves = function (i, j, chessboard) {
	const cb = chessboard;
	const turn = cb[i][j][0];
	var valid_moves = [];
	let r, c;
	// Check rules first
	switch (cb[i][j][1]) {
		// Check King and Advisory(==3)
		case "k":
			for (var k = 0; k < k_move.length; k++) {
				r = k_move[k][0] + i;
				c = k_move[k][1] + j;
				if (valids[r + shift][c + shift] == 3 && cb[r][c][0] != turn) {
					valid_moves.push([r, c]);
				}
			}
			break;
		case "a":
			for (var k = 0; k < a_move.length; k++) {
				r = a_move[k][0] + i;
				c = a_move[k][1] + j;
				if (valids[r + shift][c + shift] == 3 && cb[r][c][0] != turn) {
					valid_moves.push([r, c]);
				}
			}
			break;
		// Check Elephant(>=2)
		case "b":
			for (var k = 0; k < b_move.length; k++) {
				r = b_move[k][0] + i;
				c = b_move[k][1] + j;
				if (valids[r + shift][c + shift] >= 2 && cb[r][c][0] != turn && cb[i + (b_move[k][0] / 2)][j + (b_move[k][1] / 2)] == "--") {
					valid_moves.push([r, c]);
				}
			}
			break;
		// Check Others(>=1)
		case "n":
			for (var k = 0; k < n_move.length; k++) {
				r = n_move[k][0] + i;
				c = n_move[k][1] + j;
				if (valids[r + shift][c + shift] >= 1 && cb[r][c][0] != turn && cb[i + ((1 + n_move[k][0]) % 2)][j + ((1 + n_move[k][1]) % 2)] == "--") {
					valid_moves.push([r, c]);
				}
			}
			break;
		case "r":
			var pot_r = i;
			var pot_c = j;
			// Check vertical direction
			pot_r = i - 1;
			while (valids[pot_r + shift][pot_c + shift] >= 1 && cb[pot_r][pot_c][0] != turn) {
				valid_moves.push([pot_r, pot_c]);
				// Need stop here
				if (cb[pot_r][pot_c][0] != '-') {
					break;
				}
				pot_r--;
			}
			pot_r = i + 1;
			while (valids[pot_r + shift][pot_c + shift] >= 1 && cb[pot_r][pot_c][0] != turn) {
				valid_moves.push([pot_r, pot_c]);
				// Need stop here
				if (cb[pot_r][pot_c][0] != '-') {
					break;
				}
				pot_r++;
			}
			// Check horizontal direction
			pot_r = i;
			pot_c = j - 1;
			while (valids[pot_r + shift][pot_c + shift] >= 1 && cb[pot_r][pot_c][0] != turn) {
				valid_moves.push([pot_r, pot_c]);
				// Need stop here
				if (cb[pot_r][pot_c][0] != '-') {
					break;
				}
				pot_c--;
			}
			pot_c = j + 1;
			while (valids[pot_r + shift][pot_c + shift] >= 1 && cb[pot_r][pot_c][0] != turn) {
				valid_moves.push([pot_r, pot_c]);
				// Need stop here
				if (cb[pot_r][pot_c][0] != '-') {
					break;
				}
				pot_c++;
			}
			break;
		case 'c':
			var next_opponent_chess = function (x, y, direction_i, direction_j) {
				x += direction_i;
				y += direction_j;
				while (x >= 0 && x < 10 && y >= 0 && y < 9) {
					if (cb[x][y][0] == turn) {
						return [-1, -1];
					}
					else if (cb[x][y][0] != "-") {
						return [x, y];
					}
					x += direction_i;
					y += direction_j;
				}
				return [-1, -1];
			}

			var pot_r = i;
			var pot_c = j;
			// Check vertical direction
			pot_r = i - 1;
			while (valids[pot_r + shift][pot_c + shift]) {
				// Need stop here
				if (cb[pot_r][pot_c][0] != "-") {
					var temp = next_opponent_chess(pot_r, pot_c, -1, 0);
					if (temp[0] != -1 && temp[1] != -1) {
						valid_moves.push(temp);
					}
					break;
				}
				valid_moves.push([pot_r, pot_c]);
				pot_r--;
			}
			pot_r = i + 1;
			while (valids[pot_r + shift][pot_c + shift]) {
				// Need stop here
				if (cb[pot_r][pot_c][0] != "-") {
					var temp = next_opponent_chess(pot_r, pot_c, 1, 0);
					if (temp[0] != -1 && temp[1] != -1) {
						valid_moves.push(temp);
					}
					break;
				}
				valid_moves.push([pot_r, pot_c]);
				pot_r++;
			}

			// Check horizontal direction
			pot_r = i;
			pot_c = j - 1;
			while (valids[pot_r + shift][pot_c + shift]) {
				// Need stop here
				if (cb[pot_r][pot_c][0] != "-") {
					var temp = next_opponent_chess(pot_r, pot_c, 0, -1);
					if (temp[0] != -1 && temp[1] != -1) {
						valid_moves.push(temp);
					}
					break;
				}
				valid_moves.push([pot_r, pot_c]);
				pot_c--;
			}
			pot_c = j + 1;
			while (valids[pot_r + shift][pot_c + shift]) {
				// Need stop here
				if (cb[pot_r][pot_c][0] != "-") {
					var temp = next_opponent_chess(pot_r, pot_c, 0, 1);
					if (temp[0] != -1 && temp[1] != -1) {
						valid_moves.push(temp);
					}
					break;
				}
				valid_moves.push([pot_r, pot_c]);
				pot_c++;
			}
			break;
		case 'p':
			if (turn == "r") {
				// Checking forward movement
				if (valids[i - 1 + shift][j + shift] >= 1 && cb[i - 1][j][0] != turn) {
					valid_moves.push([i - 1, j]);
				}
				// Checking left-right movement
				if (i <= 4) {
					if (j > 0 && cb[i][j - 1][0] != turn) {
						valid_moves.push([i, j - 1]);
					}
					if (j < 8 && cb[i][j + 1][0] != turn) {
						valid_moves.push([i, j + 1]);
					}
				}
			}
			else {
				// Checking forward movement
				if (valids[i + 1 + shift][j + shift] >= 1 && cb[i + 1][j][0] != turn) {
					valid_moves.push([i + 1, j]);
				}
				// Checking left-right movement
				if (i >= 5) {
					if (j > 0 && cb[i][j - 1][0] != turn) {
						valid_moves.push([i, j - 1]);
					}
					if (j < 8 && cb[i][j + 1][0] != turn) {
						valid_moves.push([i, j + 1]);
					}
				}
			}
			break;
	}
	return valid_moves;
}

// Check if the current player is in check in the cb chessboard
module.exports.is_check = function (turn, chessboard) {
	const t = turn;
	const cb = chessboard;
	// Get King's position of current player
	var opponent = (t == "r" ? "b" : "r");
	var king = [];
	for (var i = 0; i < 10; i++) {
		for (var j = 3; j < 6; j++) {
			if (cb[i][j] == t + "k") {
				king = [i, j];
			}
		}
	}

	// Check if opponent's chess can kill the king
	for (var i = 0; i < 10; i++) {
		for (var j = 0; j < 9; j++) {
			if (cb[i][j][0] == opponent) {
				// Check special case
				if (cb[i][j][1] == "k") {
					if (j == king[1]) {
						var l = i + 1;
						var u = king[0];
						if (l > u) {
							l = king[0] + 1;
							u = i;
						}
						while (l < u && cb[l][j] == "--") {
							l++;
						}
						if (l == u) {
							return true;
						}
						break;
					}
				}
				let v = calculate_moves(i, j, cb);
				for (var k = 0; k < v.length; k++) {
					if ((king[0] == v[k][0]) && (king[1] == v[k][1])) {
						return true;
					}
				}
			}
		}
	}
	return false;
}

// Psudomove
var pseudoMove = function (i, j, valid_moves, turn, cb) {
	//Psedomove
	var ori_r = i;
	var ori_c = j;
	var t_moves = [];
	// Check the position of chessboard
	for (var i = 0; i < valid_moves.length; i++) {
		var t = cb[valid_moves[i][0]][valid_moves[i][1]];
		cb[valid_moves[i][0]][valid_moves[i][1]] = cb[ori_r][ori_c];
		cb[ori_r][ori_c] = "--";
		if (!module.exports.is_check(turn, cb)) {
			t_moves.push(valid_moves[i]);
		}
		cb[ori_r][ori_c] = cb[valid_moves[i][0]][valid_moves[i][1]];
		cb[valid_moves[i][0]][valid_moves[i][1]] = t;
	}
	return t_moves;
}

// Check checkmate
module.exports.checkmate = function (player_turn, cb) {
	var tem = [];
	for (var i = 0; i < 10; i++) {
		for (var j = 0; j < 9; j++) {
			if (cb[i][j][0] == player_turn) {
				tem = calculate_moves(i, j, cb);
				tem = pseudoMove(i, j, tem, player_turn, cb);
				if (tem.length != 0) {
					return false;
				}
			}
		}
	}
	return true;
}

//Get all valid moves
module.exports.get_valid_moves = function (row, col, turn, chessboard) {
	const r = row;
	const c = col;
	const t = turn;
	if (chessboard[r][c][0] != turn)
		return [];
	var cb = dup_chessboard(chessboard);
	var v_moves = calculate_moves(r, c, cb);
	v_moves = pseudoMove(r, c, v_moves, t, cb);
	return v_moves;
}

////Computers////
/////////////////
// Get opponent
/*
var opponent = function(s){
	if(s == 'r'){
		return 'b';
	}
	else if(s == 'b'){
		return 'r';
	}
}

// Shuffle function
var sort = function(arr) {
	var j;
	for(var i=arr.length-1; i>0; i--) {
		j = Math.floor(Math.random()*i);
		var tem = arr[i];
		arr[i] = arr[j];
		arr[j] = tem;
		}
	var index = 0;
	for(var i=0; i<arr.length; i++){
		if(chessboard[arr[i][0]][arr[i][1]][1]=='h' || chessboard[arr[i][0]][arr[i][1]][1]=='c' || chessboard[arr[i][0]][arr[i][1]][1]=='r'){
			var tem = arr[i];
			  arr[i] = arr[index];
			  arr[index] = tem;
			  index++;
		}
	}
}

// Score Sheet
var get_score = function(i, j, cb){
	switch(cb[i][j][1]){
		case 'r':
			return 90;
		case 'c':
			return 45;
		case 'h':
			return 40;
		case 's':
			if(cb[i][j][0] == 'r'){
				return rs[i][j];
			}
			else{
				return rs[9-i][j];
			}
		case 'e':
			return 20;
		case 'a':
			return 20;
		case 'k':
			return 1000;
		default:
			return 0;
	}
}

// Simple Evaluation function
var eval_score = function(cb){
	var score = 0;
	for(var i=0; i<10; i++){
		for(var j=0; j<9; j++){
			if(cb[i][j][0] == player){
				score += get_score(i, j, cb);
			}
			else if(cb[i][j][0] == opponent(player)){
				score -= get_score(i, j, cb);
			}
		}
	}
	return score;
}

// Quiescent search to prevent consecutive eating and wrong evaluation
// Genereate Captures
var generate_cap = function(max_p, cb){
	var caps = [];
	var chess_list=[];
	for(var i=0; i<10; i++){
		for(var j=0; j<9; j++){
			if(cb[i][j][0] == max_p){
				chess_list.push([i,j]);
			}
		}
	}
	for(var k=0; k<chess_list.length; k++){
		var i = chess_list[k][0];
		var j = chess_list[k][1];
		var t = calculate_moves(i, j, cb);
		t = pseudoMove(i, j, t, max_p, cb);
		var tem = [];
		for(var l=0; l<t.length; l++){
			if(cb[t[l][0]][t[l][1]] != "--"){
				tem.push(t[l]);
			}
		}
		if(tem.length != 0){
			tem.push([i,j]);
			caps.push(tem);
		}
	}
	return caps;
}
*/
// var q_search = function(max_p, a, b, cb){
// 	if(stalemate(max_p, cb)){
// 		if(max_p == player){
// 			return -INF/2;
// 		}
// 		else{
// 			return INF/2;
// 		}
// 	}
// 	var e_val = eval_score(cb);
// 	var maxVal = -INF;
// 	var minVal = INF;
// 	if(max_p == player){
// 		maxVal = Math.max(maxVal, e_val);
// 		a = Math.max(a, e_val);
// 		if(b<=a){
// 			return maxVal;
// 		}
// 	}
// 	else{
// 		minVal = Math.min(minVal, e_val);
// 		b = Math.min(b, e_val);
// 		if(b<=a){
// 			return minVal;
// 		}
// 	}
// 	var caps = generate_cap(max_p, cb);
// 	while(caps.length > 0){
// 		// Get capture moves and the corresponding chess
// 		var t = caps.pop();
// 		// Get chess location
// 		var tem = t.pop();
// 		var i = tem[0];
// 		var j = tem[1];
// 		for(var k=0; k<t.length; k++){
// 			var ori = cb[i][j];
// 			var target = cb[t[k][0]][t[k][1]];
// 			cb[t[k][0]][t[k][1]] = cb[i][j];
// 			cb[i][j] = "--";
// 			e_val = q_search(opponent(max_p), a, b, cb);
// 			cb[i][j] = ori;
// 			cb[t[k][0]][t[k][1]] = target;
// 			if(max_p == player){
// 				maxVal = Math.max(maxVal, e_val);
// 				a = Math.max(a, e_val);
// 			}
// 			else{
// 				minVal = Math.min(minVal, e_val);
// 				b = Math.min(b, e_val);
// 			}
// 			if(b<=a){
// 				break;
// 			}
// 		}
// 	}
// 	if(max_p == player){
// 		return maxVal;
// 	}
// 	else{
// 		return minVal;
// 	}
// }

// var need_q_s = false;
// var minimax = function(depth, max_p, a, b, cb){
// 	if(stalemate(max_p, cb)){
// 		if(max_p == player){
// 			return -INF/2;
// 		}
// 		else{
// 			return INF/2;
// 		}
// 	}
// 	if(depth == 0){
// 		if(need_q_s){
// 			return q_search(max_p, a, b, cb);
// 		}
// 		else{
// 			return eval_score(cb);
// 		}
// 	}
// 	var chess_list=[];
// 	for(var i=0; i<10; i++){
// 		for(var j=0; j<9; j++){
// 			if(cb[i][j][0] == max_p){
// 				chess_list.push([i,j]);
// 			}
// 		}
// 	}
// 	if(max_p == player){
// 		sort(chess_list);
// 		var maxVal = -INF;
// 		for(var x=0; x<chess_list.length; x++){
// 			var i = chess_list[x][0];
// 			var j = chess_list[x][1];
// 			var t = calculate_moves(i, j, cb);
// 			t = pseudoMove(i, j, t, max_p, cb);
// 			for(var k=0; k<t.length; k++){
// 				var ori = cb[i][j];
// 				var target = cb[t[k][0]][t[k][1]];
// 				cb[t[k][0]][t[k][1]] = cb[i][j];
// 				cb[i][j] = "--";
// 				// Enable quiescent search only if this is the last move and capture any opponent chess
// 				if(depth == 1 && target != "--"){
// 					need_q_s = true;
// 				}
// 				e_val = minimax(depth-1, opponent(max_p), a, b, cb);
// 				// Disable quiescent search
// 				if(depth == 1 && target != "--"){
// 					need_q_s = false;
// 				}
// 				cb[i][j] = ori;
// 				cb[t[k][0]][t[k][1]] = target;
// 				if(depth == maxDepth && e_val > maxVal){
// 					ori_r = i;
// 					ori_c = j;
// 					move_r = t[k][0];
// 					move_c = t[k][1];
// 				}
// 				maxVal = Math.max(maxVal, e_val);
// 				a = Math.max(a, e_val);
// 				if(b<=a){
// 					break;
// 				}
// 			}
// 		}
// 		return maxVal;
// 	}
// 	else{
// 		var minVal = INF;
// 		for(var x=0; x<chess_list.length; x++){
// 			var i = chess_list[x][0];
// 			var j = chess_list[x][1];
// 			var t = calculate_moves(i, j, cb);
// 			t = pseudoMove(i, j, t, max_p, cb);
// 			for(var k=0; k<t.length; k++){
// 				var ori = cb[i][j];
// 				var target = cb[t[k][0]][t[k][1]];
// 				cb[t[k][0]][t[k][1]] = cb[i][j];
// 				cb[i][j] = "--";
// 				// Enable quiescent search only if this is the last move and capture any opponent chess
// 				if(depth == 1 && target != "--"){
// 					need_q_s = true;
// 				}
// 				e_val = minimax(depth-1, opponent(max_p), a, b, cb);
// 				// Disable quiescent search
// 				if(depth == 1 && target != "--"){
// 					need_q_s = false;
// 				}
// 				cb[i][j] = ori;
// 				cb[t[k][0]][t[k][1]] = target;
// 				minVal = Math.min(minVal, e_val);
// 				b = Math.min(b, e_val);
// 				if(b<=a){
// 					break;
// 				}
// 			}
// 		}
// 		return minVal;
// 	}
// }

// module.exports.run = function(p, cb){
// 	ori_r = -1;
// 	ori_c = -1;
// 	move_r = -1;
// 	move_c = -1;
// 	player = p;
// 	chessboard = dup_chessboard(cb);
// 	console.log(minimax(maxDepth, player, -INF, INF,chessboard));
// 	return [ori_r, ori_c, move_r, move_c];
// }

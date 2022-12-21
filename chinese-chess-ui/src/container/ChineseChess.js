import styled from "styled-components";
import {cloneDeep} from "lodash";
import ChessUtil from '../utils/chess';
import { useState, useEffect } from "react";
// Global variables
const CHESS_SIZE = 55;
const CHESS_MAP = {
	"br": "BR",
	"rr": "RR",
	"bn": "BN",
	"rn": "RN",
	"bb": "BB",
	"rb": "RB",
	"ba": "BA",
	"ra": "RA",
	"bk": "BK",
	"rk": "RK",
	"bc": "BC",
	"rc": "RC",
	"bp": "BP",
	"rp": "RP",
	"--": "OO"
};
const imageOrder = ["BR", "RR", "BN", "RN", "BB", "RB", "BA", "RA", "BK", "RK", "BC", "RC", "BP", "RP", "--"];
const BOARD_TYPES = ["CANVAS.JPG", "DROPS.JPG", "GREEN.JPG", "SHEET.JPG", "WHITE.JPG", "WOOD.JPG"];
const CHESS_TYPES = ["COMIC", "DELICATE", "POLISH", "WOOD", "MOVESKY", "XQSTUDIO"]


const Container = styled.div`
	width: 100%;
	display: flex;
	justify-content: center;
`
const Gameboard = styled.div`
	width: 530px;
	height: 580px;
	background-image: url(${props=>props.img});
	background-position: center;
	background-size: contain;
	background-repeat: no-repeat;
`
const Chess = styled.img`
	display: inline-block;
	width: ${props=>props.size}px;
	height: ${props=>props.size}px;
`
const GameInfo = styled.div`
	font-size: 1.5rem;
	width: 375px;
	height: 580px;
`
const GameLogo = styled.img`
	margin: 0 auto;
	display: block;
	width: 80%;
`
const Button = styled.button`
	padding: 0.5rem 0.75rem;
	margin: 0.25rem;
	font-size: 1.25rem;
	outline: none;
	border: none;
	border-radius: 4px;
	color: #e3e3e3;
	background-color: #343a40;
	opacity: 0.91;
	&:hover{
		opacity: 1;
		cursor: pointer;
	}
`
const ModalContainer = styled.div`
	width: 100%;
	height: 100%;
	position: fixed;
	z-index: 10000;
	left: 0;
	top: 0;
	overflow: auto;
	background-color: rgba(0,0,0,0.4);
`
const Modal = styled.div`
	width: 500px;
	height: 700px;
	background-color: #fefefe;
	padding: 12px;
	margin: 10% auto;
	position: relative;
`
const ModalHeader = styled.div`
	font-size: 25px;
	margin: 12px;
	span {
		position: absolute;
		font-size: 25px;
		top: 8px;
		right: 12px;
		cursor: pointer;
	}
`;
const ModalBody = styled.div``;
const Row = styled.div`
	width: 100%;
	margin: 1rem;
	display: flex;
`
const Col = styled.div`
	width: 250px;
`
const ChineseChess = () => {
	const [chessboard, setChessboard] = useState([
			["br", "bn", "bb", "ba", "bk", "ba", "bb", "bn", "br"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["--", "bc", "--", "--", "--", "--", "--", "bc", "--"],
			["bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["rp", "--", "rp", "--", "rp", "--", "rp", "--", "rp"],
			["--", "rc", "--", "--", "--", "--", "--", "rc", "--"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["rr", "rn", "rb", "ra", "rk", "ra", "rb", "rn", "rr"]
	]);
	const [showConfig, setShowConfig] = useState(false);
	const [turn, setTurn] = useState("r");
	const [chessboardImg, setChessboardImg] = useState("WOOD.JPG");
	const [chessImgType, setChessImgType] = useState("WOOD");
	const [selected, setSelected] = useState([-1, -1]);
	const [isStalemate, setIsStalemate] = useState(false);
	const [validMoves, setValidMoves] = useState([]);
	const [reversed, setReversed] = useState(false);
	const [showValidMoves, setShowValidMoves] = useState(false);

	useEffect(()=>{

	}, []);

	const restartGame = ()=>{
		setTurn("r");
		setSelected([-1, -1]);
		setIsStalemate(false);
		setReversed(false);
		setShowValidMoves(false);
		setValidMoves([]);
		setChessboard([
			["br", "bn", "bb", "ba", "bk", "ba", "bb", "bn", "br"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["--", "bc", "--", "--", "--", "--", "--", "bc", "--"],
			["bp", "--", "bp", "--", "bp", "--", "bp", "--", "bp"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["rp", "--", "rp", "--", "rp", "--", "rp", "--", "rp"],
			["--", "rc", "--", "--", "--", "--", "--", "rc", "--"],
			["--", "--", "--", "--", "--", "--", "--", "--", "--"],
			["rr", "rn", "rb", "ra", "rk", "ra", "rb", "rn", "rr"]
		]);
	}

	const undo = ()=>{

	}
	const move = (o_i, o_j, i, j) => {
		let cb = cloneDeep(chessboard)
		cb[i][j] = cb[o_i][o_j];
		cb[o_i][o_j] = "--";
		setChessboard(cb);
		setSelected([-1, -1]);
		setTurn(t => t === "r" ? "b" : "r");
		setValidMoves([]);
	}
	const reverse = (isReversed) => {
		setSelected([-1, -1]);
		setChessboard(cb => cloneDeep(cb).map(r=>r.reverse()).reverse());
		setValidMoves([]);
	}
	return (
		<Container id="game_area" onClick={()=>{
				setSelected([-1, -1]);
			}}>
			<Gameboard img={`${process.env.PUBLIC_URL}/img/s/${chessboardImg}`}>
				{
					chessboard.map((row, i) =>
						(<>
							{
								row.map((chess, j) => <Chess
									key={`chess-${i}-${j}`}
									src={
										`${process.env.PUBLIC_URL}/img/s/${chessImgType}/${CHESS_MAP[chess]}${
											(i == selected[0] && j == selected[1]) ||
											(showValidMoves && validMoves.find(m => m[0] == i && m[1] == j))? "S" : ""}.GIF`
									}
									size={CHESS_SIZE}
									onClick={(e)=>{
										e.stopPropagation();
										// Check if a move
										if(selected[0] !== -1 && selected[1] !== -1 && chessboard[selected[0]][selected[1]][0] === turn){
											for(let m of validMoves){
												if(i == m[0] && j == m[1]){
													// Move
													move(selected[0], selected[1], i, j);
													return;
												}
											}
										}
										setSelected([i, j])
										let moves = ChessUtil.get_valid_moves(i, j, turn, chessboard);
										setValidMoves(moves);
									}}/>)
							}
						</>)
					)
				}
			</Gameboard>
			<GameInfo>
				<GameLogo src={`${process.env.PUBLIC_URL}/img/chess_logo.png`} alt="Chinese Chess" />
				<p className="checkmate"></p>
				<p>Player's turn: <span style={{fontSize: "1.5rem", color: turn === "r" ? "red" : "black"}}>{turn === "r" ? "RED" : "BLACK"}</span></p>
				<Button onclick={() => restartGame()}>New Game</Button>
				<Button onclick={() => undo()}>Undo</Button>
				<Button onClick={() => setShowConfig(true)}>Settings</Button>
			</GameInfo>
			<ModalContainer hidden={!showConfig} onClick={()=>setShowConfig(false)}>
				<Modal onClick={e=>e.stopPropagation()}>
					<ModalHeader>
						Settings
						<span onClick={()=>setShowConfig(false)}>&times;</span>
					</ModalHeader>
					<ModalBody>
						<Row>
							<Col>Chessboard Type</Col>
							<Col>
								<select value={chessboardImg} onChange={e=>setChessboardImg(e.target.value)}>
									{
										BOARD_TYPES.map(b => <option value={b}>{b.split(".")[0].toLowerCase()}</option>)
									}
								</select>
							</Col>
						</Row>
						<Row>
							<Col>Chess Type</Col>
							<Col>
								<select value={chessImgType} onChange={e=>setChessImgType(e.target.value)}>
									{
										CHESS_TYPES.map(c => <option value={c}>{c.toLowerCase()}</option>)
									}
								</select>
							</Col>
						</Row>
						<Row>
							<Col>Show valid moves?</Col>
							<Col><input type="checkbox" defaultChecked={showValidMoves} onChange={e=>setShowValidMoves(e.target.checked)}/></Col>
						</Row>
						<Row>
							<Col>Reverse black chess?</Col>
							<Col><input type="checkbox" defaultChecked={reversed} onChange={e=>setReversed(e.target.checked)}/></Col>
						</Row>
					</ModalBody>
			</Modal>
		</ModalContainer>
	</Container>)
}

export default ChineseChess;

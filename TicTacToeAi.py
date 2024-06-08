import copy
import random
import numpy as np
from colorama import Fore, Style
import tkinter as tk
from tkinter import messagebox
# Thay đổi kích cỡ bàn cờ tại đây
SIZE = 15
EMPTY = ' '
AI = "x"
HUMAN = "o"

class Node:
    def __init__(self, board, size, roleJustMoved, moved=None):
        self.board = board
        self.size = int(size)
        self.moved = moved  # Nước đi vừa được thực hiện
        self.roleJustMoved = roleJustMoved  # Role của người vừa thực hiện nước đi

    def getChildNodes(self):
        childNodes = []
        # Xác định vùng giới hạn
        min_i, min_j, max_i, max_j = self.get_bounding_box()
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                if self.board[i][j] == EMPTY:
                    newBoard = copy.deepcopy(self.board)
                    roleNextTurn = 'x' if self.roleJustMoved == 'o' else 'o'
                    newBoard[i][j] = roleNextTurn
                    moved = (i, j)
                    newNode = Node(newBoard, self.size, roleNextTurn, moved)
                    if shortestManhattanDistance(newNode) <= 2:
                        childNodes.append(newNode)
        return childNodes

    def get_bounding_box(self):
        min_i = max_i = min_j = max_j = -1
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != EMPTY:
                    if min_i == -1 or i < min_i: min_i = i
                    if max_i == -1 or i > max_i: max_i = i
                    if min_j == -1 or j < min_j: min_j = j
                    if max_j == -1 or j > max_j: max_j = j
        min_i = max(0, min_i - 2)
        max_i = min(self.size - 1, max_i + 2)
        min_j = max(0, min_j - 2)
        max_j = min(self.size - 1, max_j + 2)
        return min_i, min_j, max_i, max_j

    def shortestManhattanDistance(self):
        return shortestManhattanDistance(self)

    def pr(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == ' ':
                    print("-", end=" ")
                else:
                    print(self.board[i][j], end=" ")
            print()

def manhattanDistance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def shortestManhattanDistance(node: Node):
    min_distance = 1000
    for i in range(node.size):
        for j in range(node.size):
            if node.board[i][j] != EMPTY and (i, j) != node.moved:
                distance = manhattanDistance(node.moved, (i, j))
                if distance < min_distance:
                    min_distance = distance
                    if min_distance <= 2:
                        return 2
    return min_distance

def get_move(board, size):
    isFirstMove = True
    for row in board:
        for cell in row:
            if cell != ' ': 
                isFirstMove = False
                break
    if isFirstMove: return ( (size // 2), (size // 2) )
    maxVal = -np.inf
    maxChilds = []
    node = Node(board, size, HUMAN)  # Đổi vai XO tại đây
    childNodes = node.getChildNodes()

    if not childNodes:
        return None

    for child in childNodes:
        val = minimax(child, 0, False)
        if maxVal < val:
            maxVal = val
            maxChilds.clear()
            maxChilds.append(child)
        elif maxVal == val:
            maxChilds.append(child)
    myMove = random.choice(maxChilds).moved
    return myMove

'''
def get_move(board, size):
    minVal = np.inf
    minChilds = []
    node = Node(board, size, HUMAN)  # Đổi vai XO tại đây
    # Find all available positions on the board
    childNodes = node.getChildNodes()

    # If there are no available moves, return None
    if not childNodes:
        return None

    # Chọn nước đi có điểm thấp nhất
    for childNode in childNodes:
        childNode.pr()
        val = minimax(childNode, 0, True)
        print("val: ", val)
        if minVal > val:
            minVal = val
            minChilds.clear()
            minChilds.append(childNode)
        elif minVal == val:
            minChilds.append(childNode)
    myMove = random.choice(minChilds).moved  # Ngẫu nhiên trong số các nước đi có điểm min
    return myMove
'''
def minimax(node: Node, depth, maximizingPlayer):
    return alphabeta(node, depth, -np.inf, np.inf, maximizingPlayer)

def alphabeta(node: Node, depth, alpha, beta, maximizingPlayer: bool):
    if isGameOver(node) or depth == 0:
        return value(node)
    if maximizingPlayer:
        maxEval = -np.inf
        for childNode in node.getChildNodes():
            eval = alphabeta(childNode, depth - 1, alpha, beta, False)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = np.inf
        for childNode in node.getChildNodes():
            eval = alphabeta(childNode, depth - 1, alpha, beta, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval

def isGameOver(node: Node):
    return checkWin(node, AI) or checkWin(node, HUMAN)

# Constants
attackReward = [0, 10, 1000, 2000, 17500, 100000]
defendReward = [0, 9, 3499, 3999, 18999, 99999]

BONUS_NON_BLOCK_BOTH_ENDS_ATTACK = 4000
BONUS_NON_BLOCK_BOTH_ENDS_DEFEND = 12000
BONUS_OPEN_ATTACK = 1000
BONUS_AVOID_OPPONENT_WRAP = 2000
BONUS_EXPAND_SPACE = 1000
BONUS_ATTACK_CHAIN = 1000
BONUS_BLOCK_ONE_END_ATTACK = -1750
BONUS_BLOCK_ONE_END_DEFEND = -2000
MIN_ATTACK_THRESHOLD = 4000
BONUS_OPEN_SPACE = 2000
BONUS_BLOCK_ENEMY_OPEN_SPACE = 2000
BONUS_DIAGONAL_ATTACK = 1000
BONUS_OPENING_THREE = 2000
BONUS_SURE_DEFENSE = 75000
BONUS_SURE_ATTACK = 90000
def evaluate_line(node, role, dx, dy):
    (i, j) = node.moved
    b = node.board
    currentRole = node.roleJustMoved if role == 'att' else (HUMAN if node.roleJustMoved == AI else AI)
    oppRole = HUMAN if currentRole == AI else AI

    val = 0
    currPieces = 0
    empty_count = 0
    block_start = block_end = False
    open_space = 0
    check_open_space = 0
    attack_chain = 0
    nearest_opponent_distance = float('inf')
    nearest_opponent_sum = 0  # Tổng số quân của đối thủ gần nhất
    max_attack_chain = 0
    open_space_forward = 0
    open_space_backward = 0
    # Forward direction
    if i < 1 or i >= node.size - 1 or j < 1 or j >= node.size - 1:
        val -= 3000 # Trừ điểm nếu nước đi gần cạnh bàn cờ
    count = 1
    while count < 5 and 0 <= i + count * dx < node.size and 0 <= j + count * dy < node.size:
        if b[i + count * dx][j + count * dy] == currentRole: 
            if nearest_opponent_sum == 0:
                nearest_opponent_sum = 1 
            check_open_space = 1
            currPieces += 1
            attack_chain += 1
            max_attack_chain = max(max_attack_chain, attack_chain)
        elif b[i + count * dx][j + count * dy] == EMPTY:
            empty_count += 1
            if check_open_space == 0:
                open_space += 1
                open_space_forward += 1
            if attack_chain >= 2 and role == 'att':
                val += BONUS_ATTACK_CHAIN
            attack_chain = 0
        else:
            if(empty_count < 1):
                block_end = True
            break
        count += 1
    attack_chain = 0
    # Backward direction
    count = 1
    check_open_space = 0
    empty_count = 0
    while count < 5 and 0 <= i - count * dx < node.size and 0 <= j - count * dy < node.size:
        if b[i - count * dx][j - count * dy] == currentRole: 
            if nearest_opponent_sum == 0:
                nearest_opponent_sum = 1 
            elif nearest_opponent_sum == 1:
                nearest_opponent_sum = 2
            currPieces += 1
            attack_chain += 1
            check_open_space = 1
            max_attack_chain = max(max_attack_chain, attack_chain)
        elif b[i - count * dx][j - count * dy] == EMPTY:
            empty_count += 1
            if check_open_space == 0:
                open_space += 1
                open_space_backward += 1
            if attack_chain >= 2 and role == 'att':
                val += BONUS_ATTACK_CHAIN
            attack_chain = 0
        else:
            if(empty_count < 1):
                block_start = True
            break
        count += 1
    nearest_opponent_distance = min(open_space_backward + 1, open_space_forward + 1)
    if currPieces == 0:
        return 0

    # Tính điểm thưởng cho các trường hợp tấn công và phòng thủ
    if role == 'att' and open_space >= 2:
        val += BONUS_OPEN_ATTACK  # Ưu tiên nước tấn công mở đẹp

    if role == 'def' and open_space >= 2:
        val += BONUS_AVOID_OPPONENT_WRAP  # Hạn chế nước đôi của đối thủ

    # Thưởng mở rộng không gian
    if open_space >= 2:
        val += BONUS_EXPAND_SPACE

    if not block_start and not block_end:
        if currPieces >= 3:
            val += BONUS_NON_BLOCK_BOTH_ENDS_ATTACK if role == 'att' else BONUS_NON_BLOCK_BOTH_ENDS_DEFEND
        if currPieces == 2 and (open_space <= 1 or (max_attack_chain >= 2 and nearest_opponent_distance <= 2)):
            val += (BONUS_NON_BLOCK_BOTH_ENDS_ATTACK / 2) if role == 'att' else (BONUS_NON_BLOCK_BOTH_ENDS_DEFEND / 1.75)

    

    if empty_count > 0:
        if role == 'att':
            val += attackReward[currPieces]  # Có thể tấn công cách 1 ô

    if block_start and block_end:
        if role == 'att':
            val += BONUS_BLOCK_ONE_END_ATTACK * 5
        else:
            val += BONUS_BLOCK_ONE_END_DEFEND * 5
    elif block_start or block_end:
        if role == 'att':
            val += BONUS_BLOCK_ONE_END_ATTACK *2.25
        else:
            val += BONUS_BLOCK_ONE_END_DEFEND *3 

    # Kiểm tra nếu nước đi là nước tấn công chéo
    if dx != 0 and dy != 0:
        val += BONUS_DIAGONAL_ATTACK

    if currPieces == 3 and not block_start and not block_end:
        if role == 'att' and empty_count > 0:  # Nếu là nước tấn công và còn cách 1 ô trống
            val += BONUS_OPENING_THREE

    if currPieces == 3 and nearest_opponent_distance == 1 and not (block_start and block_end) and ((open_space == 0 and max_attack_chain >=2) or (max_attack_chain >=3 and nearest_opponent_distance == 1)) :
        if role == 'def':
            val += BONUS_SURE_DEFENSE / 3.75
        if role == 'att':
            val += BONUS_SURE_ATTACK / 2.5
    if currPieces >= 4 and nearest_opponent_distance == 1 and max_attack_chain >=2 and open_space != 1:
        if role == 'def':
            val += BONUS_SURE_DEFENSE
        else:
            val += BONUS_SURE_ATTACK
    if nearest_opponent_distance == 1:
        if role == 'att':
            val += BONUS_ATTACK_CHAIN
        else:
            val -= BONUS_ATTACK_CHAIN

    if nearest_opponent_sum > 1:
        val += nearest_opponent_sum * 1000 
    if currPieces > 0:
        val += currPieces * attackReward[currPieces] if role == 'att' else currPieces * defendReward[currPieces]
    # Kiểm tra điểm tấn công
    if val < MIN_ATTACK_THRESHOLD:
        if role == 'att' and open_space >= 2:
            val += BONUS_OPEN_SPACE  # Ưu tiên nước mở cờ
        elif role == 'def' and open_space >= 2:
            val += BONUS_BLOCK_ENEMY_OPEN_SPACE  # Ưu tiên nước chặn vùng thông thoáng của đối phương

    return val

def attValRow(node: Node):
    return evaluate_line(node, 'att', 0, 1)

def attValCol(node: Node):
    return evaluate_line(node, 'att', 1, 0)

def attValDiag1(node: Node):
    return evaluate_line(node, 'att', 1, 1)

def attValDiag2(node: Node):
    return evaluate_line(node, 'att', 1, -1)

def defValRow(node: Node):
    return evaluate_line(node, 'def', 0, 1)

def defValCol(node: Node):
    return evaluate_line(node, 'def', 1, 0)

def defValDiag1(node: Node):
    return evaluate_line(node, 'def', 1, 1)

def defValDiag2(node: Node):
    return evaluate_line(node, 'def', 1, -1)

def value(node: Node):
    attackVal = attValRow(node) + attValCol(node) + attValDiag1(node) + attValDiag2(node)
    defendVal = defValRow(node) + defValCol(node) + defValDiag1(node) + defValDiag2(node)
    print("Attack: {}, Defend: {}".format(attackVal, defendVal))
    return attackVal + defendVal

def checkWin(node: Node, player: str):
    return any(checkLine(node, player, direction) for direction in [(1, 0), (0, 1), (1, 1), (1, -1)])

def checkLine(node, player, direction):
    for i in range(node.size):
        for j in range(node.size):
            if node.board[i][j] == player:
                if checkDirection(node.board, player, i, j, direction):
                    return True
    return False
def check_win(board, player):
    size = len(board)
    for i in range(size):
        for j in range(size):
            if board[i][j] == player:
                # Check horizontal
                if j + 4 < size and all(board[i][j+k] == player for k in range(5)):
                    return True
                # Check vertical
                if i + 4 < size and all(board[i+k][j] == player for k in range(5)):
                    return True
                # Check diagonal \
                if i + 4 < size and j + 4 < size and all(board[i+k][j+k] == player for k in range(5)):
                    return True
                # Check diagonal /
                if i + 4 < size and j - 4 >= 0 and all(board[i+k][j-k] == player for k in range(5)):
                    return True
    return False

def check_draw(board_state):
    return all([cell != ' ' for row in board_state for cell in row])
def checkDirection(board, player, start_i, start_j, direction):
    count = 0
    for k in range(5):
        i, j = start_i + k * direction[0], start_j + k * direction[1]
        if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == player:
            count += 1
        else:
            break
    return count == 5

def print_board(board):
    print("  ", end="")
    for i in range(len(board[0])):
        print(i % 10, end=" ")
    print()
    for i, row in enumerate(board):
        print(i % 10, end=" ")
        for cell in row:
            if cell == AI:
                print(Fore.RED + cell + Style.RESET_ALL, end=" ")
            elif cell == HUMAN:
                print(Fore.GREEN + cell + Style.RESET_ALL, end=" ")
            else:
                print(cell, end=" ")
        print()
class CaroGame:
    def __init__(self, size=15):
        self.size = size
        self.board = [[EMPTY for _ in range(size)] for _ in range(size)]
        self.current_turn = HUMAN

    def reset(self):
        self.board = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.current_turn = HUMAN

    def make_move(self, row, col):
        if self.board[row][col] == EMPTY:
            self.board[row][col] = self.current_turn
            if checkWin(Node(self.board, self.size, self.current_turn, (row, col)), self.current_turn):
                return True
            self.current_turn = AI if self.current_turn == HUMAN else HUMAN
            return False
        return False

    def ai_move(self):
        move = get_move(self.board, self.size)
        if move:
            row, col = move
            self.board[row][col] = AI
            if checkWin(Node(self.board, self.size, AI, move), AI):
                return True
            self.current_turn = HUMAN
        return False

def draw_board(canvas, game):
    canvas.delete("all")
    for i in range(game.size):
        for j in range(game.size):
            x1, y1 = j * 30, i * 30
            x2, y2 = x1 + 30, y1 + 30
            canvas.create_rectangle(x1, y1, x2, y2, outline="black")
            if game.board[i][j] == HUMAN:
                canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="black")
            elif game.board[i][j] == AI:
                canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="white")

def click(event, game, canvas):
    if game.current_turn == HUMAN:
        row, col = event.y // 30, event.x // 30
        if row < game.size and col < game.size and game.board[row][col] == EMPTY:
            if game.make_move(row, col):
                messagebox.showinfo("Game Over", "Human wins!")
                game.reset()
            else:
                if game.ai_move():
                    messagebox.showinfo("Game Over", "AI wins!")
                    game.reset()
            draw_board(canvas, game)

def main():
    game = CaroGame(SIZE)
    root = tk.Tk()
    root.title("Caro Game")
    canvas = tk.Canvas(root, width=SIZE * 30, height=SIZE * 30)
    canvas.pack()
    canvas.bind("<Button-1>", lambda event: click(event, game, canvas))
    draw_board(canvas, game)
    root.mainloop()

if __name__ == "__main__":
    main()

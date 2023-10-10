from solve_starter import *

if __name__ == "__main__":
    board = read_from_file('moderate_difficults.txt')
    board.display()
    
    path, cost = a_star(board, heuristic_advanced)
    print(cost)
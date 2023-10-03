from solve_starter import *

if __name__ == "__main__":
    board = read_from_file('test.txt')
    board.display()
    
    path, cost = dfs(board)
    print(cost)
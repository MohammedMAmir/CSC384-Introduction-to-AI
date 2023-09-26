from solve_starter import *

if __name__ == "__main__":
    board = read_from_file('test.txt')
    board.display()
    state = State(board, 1, 0, 0, parent = None)
    successors = get_successors(state)
    print(len(successors))

    for state in successors:
        state.board.display()
    
############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.1
##
## Changes: 
## v1.1: removed the hfn paramete from dfs. Updated solve_puzzle() accordingly.
############################################################

from typing import List
import heapq
from heapq import heappush, heappop
import time
import argparse
import math # for infinity
import copy

from board import *

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """
    complete = True
    available_spots = state.board.storage.copy()
    for box in state.board.boxes:
        for spot in available_spots:
            if spot[0] == box[0] and spot[1] == box[1]:
                available_spots.remove(spot)
                break

    if len(available_spots) != 0:
        return False
    else:
        return True
    
    #raise NotImplementedError


def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """
    path = []
    path.insert(0,state)

    currState = state
    while(currState.depth > 0):
        currState = currState.parent
        path.insert(0,currState)
    return path

def is_obstacle(location, state):
    for obstacle in state.board.obstacles:
        if obstacle[0] == location[0] and obstacle[1] == location[1]:
            return True
    
    return False

def is_box(location, state):
    for box in state.board.boxes:
        if box[0] == location[0] and box[1] == location[1]:
            return True
    
    return False

def is_robot(location, state):
    for robot in state.board.robots:
        if robot[0] == location[0] and robot[1] == location[1]:
            return True

    return False

def copy_board(board: Board):
    new_board = Board(board.name, board.width, board.height, copy.deepcopy(board.robots), 
                      copy.deepcopy(board.boxes), copy.deepcopy(board.storage), copy.deepcopy(board.obstacles))
    return new_board

def copy_state(state: State):
    new_state = State(copy_board(state.board), state.hfn, state.f, state.depth, parent=None)
    return new_state

def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """
    temp_state = copy_state(state)
    state_list = []
    temp_state.depth = state.depth + 1
    temp_state.parent = state
    for robIndex in range(len(temp_state.board.robots)):
        #see if robot can move up:
        robot = temp_state.board.robots[robIndex]
        if is_obstacle((robot[0], robot[1]+1), temp_state) == False and is_robot((robot[0], robot[1]+1), temp_state) == False:
            if is_box((robot[0], robot[1]+1), temp_state) == True:
                #The case where there is a box in the robots path but no obstacle preventing them from pushing it
                if(is_obstacle((robot[0], robot[1]+2), temp_state) == False and 
                is_box((robot[0], robot[1]+2), temp_state) == False and
                is_robot((robot[0], robot[1]+2), temp_state) == False):
                    for boxIndex in range(len(temp_state.board.boxes)):
                        if(temp_state.board.boxes[boxIndex][0] == robot[0] and 
                        temp_state.board.boxes[boxIndex][1] == robot[1]+1):
                            listBox = list(temp_state.board.boxes[boxIndex])
                            listBox[1] += 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            listRobot = list(temp_state.board.robots[robIndex])
                            listRobot[1] += 1
                            temp_state.board.robots[robIndex] = tuple(listRobot)
                            state_list.append(copy_state(temp_state))

                            #Reset temp state
                            listRobot[1] -= 1
                            listBox[1] -= 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            temp_state.board.robots[robIndex] = tuple(listRobot)
            #The case where there is no box or obstacle obstructing the robots path            
            else:
                listRobot = list(temp_state.board.robots[robIndex])
                listRobot[1] += 1
                temp_state.board.robots[robIndex] = tuple(listRobot)
                state_list.append(copy_state(temp_state))

                #Reset temp state
                listRobot[1] -= 1
                temp_state.board.robots[robIndex] = tuple(listRobot)
    
        #see if robot can move down:
        if is_obstacle((robot[0], robot[1]-1), temp_state) == False and is_robot((robot[0], robot[1]-1), temp_state) == False:
            if is_box((robot[0], robot[1]-1), temp_state) == True:
                #The case where there is a box in the robots path but no obstacle preventing them from pushing it
                if(is_obstacle((robot[0], robot[1]-2), temp_state) == False and 
                is_box((robot[0], robot[1]-2), temp_state) == False and
                is_robot((robot[0], robot[1]-2), temp_state) == False):
                    for boxIndex in range(len(temp_state.board.boxes)):
                        if(temp_state.board.boxes[boxIndex][0] == robot[0] and 
                        temp_state.board.boxes[boxIndex][1] == robot[1]-1):
                            listBox = list(temp_state.board.boxes[boxIndex])
                            listBox[1] -= 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            listRobot = list(temp_state.board.robots[robIndex])
                            listRobot[1] -= 1
                            temp_state.board.robots[robIndex] = tuple(listRobot)
                            state_list.append(copy_state(temp_state))

                            #Reset temp state
                            listRobot[1] += 1
                            listBox[1] += 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            temp_state.board.robots[robIndex] = tuple(listRobot)
                    
            #The case where there is no box or obstacle obstructing the robots path            
            else:
                listRobot = list(temp_state.board.robots[robIndex])
                listRobot[1] -= 1
                temp_state.board.robots[robIndex] = tuple(listRobot)
                state_list.append(copy_state(temp_state))

                #Reset temp state
                listRobot[1] += 1
                temp_state.board.robots[robIndex] = tuple(listRobot)
    

        #see if robot can move left:
        if is_obstacle((robot[0]-1, robot[1]), temp_state) == False and is_robot((robot[0]-1, robot[1]), temp_state) == False:
            if is_box((robot[0]-1, robot[1]), temp_state) == True:
                #The case where there is a box in the robots path but no obstacle preventing them from pushing it
                if(is_obstacle((robot[0]-2, robot[1]), temp_state) == False and 
                is_box((robot[0]-2, robot[1]), temp_state) == False and
                is_robot((robot[0]-2, robot[1]), temp_state) == False):
                    for boxIndex in range(len(temp_state.board.boxes)):
                        if (temp_state.board.boxes[boxIndex][0] == robot[0]-1 and 
                        temp_state.board.boxes[boxIndex][1] == robot[1]):
                            listBox = list(temp_state.board.boxes[boxIndex])
                            listBox[0] -= 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            listRobot = list(temp_state.board.robots[robIndex])
                            listRobot[0] -= 1
                            temp_state.board.robots[robIndex] = tuple(listRobot)
                            state_list.append(copy_state(temp_state))

                            #Reset temp state
                            listRobot[0] += 1
                            listBox[0] += 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            temp_state.board.robots[robIndex] = tuple(listRobot)
 
            #The case where there is no box or obstacle obstructing the robots path            
            else:
                listRobot = list(temp_state.board.robots[robIndex])
                listRobot[0] -= 1
                temp_state.board.robots[robIndex] = tuple(listRobot)
                state_list.append(copy_state(temp_state))

                #Reset temp state
                listRobot[0] += 1
                temp_state.board.robots[robIndex] = tuple(listRobot)


        #see if robot can move right:
        if is_obstacle((robot[0]+1, robot[1]), temp_state) == False and is_robot((robot[0]+1, robot[1]), temp_state) == False:
            if is_box((robot[0]+1, robot[1]), temp_state) == True:
                #The case where there is a box in the robots path but no obstacle preventing them from pushing it
                if(is_obstacle((robot[0]+2, robot[1]), temp_state) == False and 
                is_box((robot[0]+2, robot[1]), temp_state) == False and
                is_robot((robot[0]+2, robot[1]), temp_state) == False):
                    for boxIndex in range(len(temp_state.board.boxes)):
                        if (temp_state.board.boxes[boxIndex][0] == robot[0]+1 and 
                        temp_state.board.boxes[boxIndex][1] == robot[1]):
                            listBox = list(temp_state.board.boxes[boxIndex])
                            listBox[0] += 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            listRobot = list(temp_state.board.robots[robIndex])
                            listRobot[0] += 1
                            temp_state.board.robots[robIndex] = tuple(listRobot)
                            state_list.append(copy_state(temp_state))

                            #Reset temp state
                            listRobot[0] -= 1
                            listBox[0] -= 1
                            temp_state.board.boxes[boxIndex] = tuple(listBox)
                            temp_state.board.robots[robIndex] = tuple(listRobot)
            #The case where there is no box or obstacle obstructing the robots path            
            else:
                listRobot = list(temp_state.board.robots[robIndex])
                listRobot[0] += 1
                temp_state.board.robots[robIndex] = tuple(listRobot)
                state_list.append(copy_state(temp_state))

                #Reset temp state
                listRobot[0] -= 1
                temp_state.board.robots[robIndex] = tuple(listRobot)

    

    return state_list
    raise NotImplementedError

def multi_prune(state, visited):
    for i in visited:
        if i == state.id:
            return True

    return False

def moveState(direction, pos_Neg, temp_State, robIndex):
    return

def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    intialState = State(init_board, 0, 0, 0, None)
    frontier = []
    visited = []
    visited.append(intialState.id)
    frontier.append(intialState)
    count = 0
    while len(frontier) > 0 and count < 100000:
        currState = frontier.pop()
        ##currState.board.display()
        if is_goal(currState):
            return get_path(currState), len(get_path(currState))
        else:
            successors_temp = get_successors(currState)
            successors_prune = []
            for succ_index in range(len(successors_temp)):
                if multi_prune(successors_temp[succ_index], visited) == False:
                    successors_prune.append(successors_temp[succ_index])
                    visited.append(successors_temp[succ_index].id)
            if len(successors_prune) != 0:
                frontier.extend(successors_prune)
        count+=1
        if count == 9999:
            currState.board.display()

    return [], -1
    raise NotImplementedError


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    raise NotImplementedError


def heuristic_basic(board):
    """
    Returns the heuristic value for the given board
    based on the Manhattan Distance Heuristic function.

    Returns the sum of the Manhattan distances between each box 
    and its closest storage point.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError


def heuristic_advanced(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError


def solve_puzzle(board: Board, algorithm: str, hfn):
    """
    Solve the given puzzle using the given type of algorithm.

    :param algorithm: the search algorithm
    :type algorithm: str
    :param hfn: The heuristic function
    :type hfn: Optional[Heuristic]

    :return: the path from the initial state to the goal state
    :rtype: List[State]
    """

    print("Initial board")
    board.display()

    time_start = time.time()

    if algorithm == 'a_star':
        print("Executing A* search")
        path, step = a_star(board, hfn)
    elif algorithm == 'dfs':
        print("Executing DFS")
        path, step = dfs(board)
    else:
        raise NotImplementedError

    time_end = time.time()
    time_elapsed = time_end - time_start

    if not path:

        print('No solution for this puzzle')
        return []

    else:

        print('Goal state found: ')
        path[-1].board.display()

        print('Solution is: ')

        counter = 0
        while counter < len(path):
            print(counter + 1)
            path[counter].board.display()
            print()
            counter += 1

        print('Solution cost: {}'.format(step))
        print('Time taken: {:.2f}s'.format(time_elapsed))

        return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The file that contains the solution to the puzzle."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['a_star', 'dfs'],
        help="The searching algorithm."
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=None,
        choices=['zero', 'basic', 'advanced'],
        help="The heuristic used for any heuristic search."
    )
    args = parser.parse_args()

    # set the heuristic function
    heuristic = heuristic_zero
    if args.heuristic == 'basic':
        heuristic = heuristic_basic
    elif args.heuristic == 'advanced':
        heuristic = heuristic_advanced

    # read the boards from the file
    board = read_from_file(args.inputfile)

    # solve the puzzles
    path = solve_puzzle(board, args.algorithm, heuristic)

    # save solution in output file
    outputfile = open(args.outputfile, "w")
    counter = 1
    for state in path:
        print(counter, file=outputfile)
        print(state.board, file=outputfile)
        counter += 1
    outputfile.close()
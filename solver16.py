#!/usr/bin/env python
# Author: Hardik Rakholiya

# solver16.py : Solves a random 15-puzzle board up to 30 moves (In 1 move multiple tiles can be sled all in the same direction)

# Solving 15-puzzle is NP-Hard problem. So there is no optimal solution for reaching the gaol state in minimum number
#  of moves. But A* algorithm still gives the efficient solution by searching through promising successors of a node,
#  and then promising successors of the successors and so on. The key here is to select the most promising successor by
#  comparing manhattan priority and linear conflicts so that with each move we are gradually moving towards the goal state.

# State space : Any possible arrangement of tiles obtained by sliding a tile in horizontally or vertically starting
#  from the initial board. These are total of 16!/2 states.

# Goal state: The state where all the tiles are in the increasing order up to 15th, followed by the empty tile
# Goal board can be written as [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]

# The cost of going from any node to its successor here is constant and is equal to 1 move (this include multiple
# tiles all sled in same direction)

# successor function : Each board has 6 successors. Successor states can be reached by moving 1 to 3 tiles vertically
# or horizontally all in same direction

# Heuristic function here is the sum of (manhattan distances)/3 and 2*(linear conflicts on horizontal or vertical
# directions)
# (manhattan distances)/3 is the lower bound of cost for solving any board if 3 tiles can move in 1 move
#  unhindered to its goal position. we are also adding linear conflict cost to priority function here since it takes
# minimum 2 extra moves to put a pair of tiles already in their goal row or columns but in reversed order(the source
# cited in code comments). Lesser the priority the more promising the state is. Using heapq we pick the successor with
# the least priority and explore its successors.


import sys
import copy
import heapq

goal_tiles = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
goal_cols = [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 0]]


# manhattan distance is the sum of vertical and horizontal distances of all the
# misplaced tiles from its goal position
def calc_manhattan_distance(tiles):
    manhattan_distance = 0
    for r in range(0, 4):
        for c in range(0, 4):
            manhattan_distance += distance_to_goal_position(tiles, r, c)

    return manhattan_distance


# vertical and horizontal distance of a tile from its goal position
def distance_to_goal_position(tiles, r, c):
    if tiles[r][c] is 0:
        return 0
    else:
        return abs((tiles[r][c] - 1) / 4 - r) + abs((tiles[r][c] - 1) % 4 - c)


def calc_horizontal_conflicts(tiles):
    conflicts = 0
    for r in range(0, 4):
        for c in range(0, 4):
            if tiles[r][c] is not 0 and tiles[r][c] in goal_tiles[r]:
                for c1 in range(c + 1, 4):
                    if tiles[r][c1] is not 0 and tiles[r][c1] in goal_tiles[r] \
                            and tiles[r][c] > tiles[r][c1]:
                        conflicts += 2
    return conflicts


def calc_vertical_conflicts(tiles):
    conflicts = 0
    for c in range(0, 4):
        for r in range(0, 4):
            if tiles[r][c] is not 0 and tiles[r][c] in goal_cols[c]:
                for r1 in range(r + 1, 4):
                    if tiles[r1][c] is not 0 and tiles[r1][c] in goal_cols[c] \
                            and tiles[r][c] > tiles[r1][c]:
                        conflicts += 2
    return conflicts


def swap_and_get_new_tiles(tiles, r1, c1, r2, c2):
    new_tiles = copy.deepcopy(tiles)
    new_tiles[r1][c1], new_tiles[r2][c2] = new_tiles[r2][c2], new_tiles[r1][c1]
    return new_tiles


def diff_manhattan_dist_on_row(tiles1, tiles2, r0):
    delta_md = 0
    for c in range(0, 4):
        if tiles1[r0][c] != tiles2[r0][c]:
            delta_md += distance_to_goal_position(tiles1, r0, c) - distance_to_goal_position(tiles2, r0, c)
    return delta_md


def diff_manhattan_dist_on_col(tiles1, tiles2, c0):
    delta_md = 0
    for r in range(0, 4):
        if tiles1[r][c0] != tiles2[r][c0]:
            delta_md += distance_to_goal_position(tiles1, r, c0) - distance_to_goal_position(tiles2, r, c0)
    return delta_md


class Board:
    def __init__(self, prev, tiles, move, delta_md):
        self.prev = prev
        self.tiles = tiles
        if prev is None:
            self.cost = 0
            self.manhattan_distance = calc_manhattan_distance(tiles)
        else:
            self.cost = prev.cost + 1
            # Because successors and previous board are different for only 1 column or row, calculate difference of
            # manhattan distances of that particular row or column and add the difference to parent's manhattan
            # distance to get manhattan distance for successors.
            # The idea comes from http://coursera.cs.princeton.edu/algs4/checklists/8puzzle.html where it says for
            # single moves the difference in manhattan distance between a board and its neighbor is either -1 or +1.
            self.manhattan_distance = prev.manhattan_distance + delta_md
        self.move = move

        # the idea of using linear conflicts comes from Algorithms course on Coursera
        # http://coursera.cs.princeton.edu/algs4/checklists/8puzzle.html
        self.linear_conflicts = calc_horizontal_conflicts(tiles) + calc_vertical_conflicts(tiles)
        self.priority = self.cost + self.manhattan_distance + self.linear_conflicts
        self.successors = []
        self.r0 = 0
        self.c0 = 0

    def get_successors(self):

        if not self.successors:
            # find the coordinates of the empty tiles first
            for r in range(0, 4):
                for c in range(0, 4):
                    if self.tiles[r][c] is 0:
                        self.r0 = r
                        self.c0 = c
                        break

            self.slide_down_and_add_successors(self.tiles, self.r0)
            self.slide_up_and_add_successors(self.tiles, self.r0)
            self.slide_right_and_add_successors(self.tiles, self.c0)
            self.slide_left_and_add_successors(self.tiles, self.c0)

        return self.successors

    def slide_down_and_add_successors(self, tiles, r):
        if r - 1 >= 0:
            new_tiles = swap_and_get_new_tiles(tiles, r, self.c0, r - 1, self.c0)
            successor = Board(self, new_tiles, 'D{}{}'.format(self.r0 - r + 1, self.c0 + 1),
                              diff_manhattan_dist_on_col(new_tiles, self.tiles, self.c0))
            self.successors.append(successor)
            self.slide_down_and_add_successors(new_tiles, r - 1)

    def slide_up_and_add_successors(self, tiles, r):
        if r + 1 < 4:
            new_tiles = swap_and_get_new_tiles(tiles, r, self.c0, r + 1, self.c0)
            successor = Board(self, new_tiles, 'U{}{}'.format(r - self.r0 + 1, self.c0 + 1),
                              diff_manhattan_dist_on_col(new_tiles, self.tiles, self.c0))
            self.successors.append(successor)
            self.slide_up_and_add_successors(new_tiles, r + 1)

    def slide_right_and_add_successors(self, tiles, c):
        if c - 1 >= 0:
            new_tiles = swap_and_get_new_tiles(tiles, self.r0, c, self.r0, c - 1)
            successor = Board(self, new_tiles, 'R{}{}'.format(self.c0 - c + 1, self.r0 + 1),
                              diff_manhattan_dist_on_row(new_tiles, self.tiles, self.r0))
            self.successors.append(successor)
            self.slide_right_and_add_successors(new_tiles, c - 1)

    def slide_left_and_add_successors(self, tiles, c):
        if c + 1 < 4:
            new_tiles = swap_and_get_new_tiles(tiles, self.r0, c, self.r0, c + 1)
            successor = Board(self, new_tiles, 'L{}{}'.format(c - self.c0 + 1, self.r0 + 1),
                              diff_manhattan_dist_on_row(new_tiles, self.tiles, self.r0))
            self.successors.append(successor)
            self.slide_left_and_add_successors(new_tiles, c + 1)

    def solution(self):
        ptr = self
        stack = []
        while ptr.prev is not None:
            stack.append(ptr.move)
            ptr = ptr.prev
        stack.reverse()
        return ' '.join(move for move in stack)

    def solution_with_info(self):
        ptr = self
        stack = []
        while ptr is not None:
            stack.append(ptr)
            ptr = ptr.prev
        stack.reverse()
        return '\n'.join(printable(board) for board in stack)

    def same_as(self, other):
        return self.tiles == other.tiles


# read the input file for the input board
def read_file(file_path):
    file = open(file_path, 'r')
    board = []
    for line in file:
        row = []
        for num in line.split():
            row.append(int(num))
        board.append(row)

    return board


# get board in a readable format
def printable(board):
    tiles_str = ''

    for row in board.tiles:
        for col in row:
            if col == 0:
                tiles_str += ' _ '
            else:
                tiles_str += '{:>2} '.format(col)
        tiles_str += '\n'

    tiles_str += 'Cost : {} Priority : {}, MD : {}, LC : {} Move : {}\n' \
        .format(board.cost, board.priority, board.manhattan_distance, board.linear_conflicts, board.move)

    return tiles_str


# check if the tiles are in the goal position
def is_goal(board):
    return board.tiles == goal_tiles


# we can check if the puzzle is solvable or not in O(n^2) time using the method listed in
# http://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/
def is_solvable(board):
    flat_board = []
    r0 = 0
    for r in range(0, 4):
        for c in range(0, 4):
            flat_board.append(board.tiles[r][c])
            if board.tiles[r][c] is 0:
                r0 = r

    inversions = 0
    for i in range(0, 16):
        for j in range(i + 1, 16):
            if flat_board[i] is not 0 and flat_board[j] is not 0 and flat_board[i] > flat_board[j]:
                inversions += 1

    if (r0 + inversions) % 2 is 1:
        return True
    else:
        return False


# solve the board
def solve(board):
    fringe = []
    heapq.heappush(fringe, (board.priority, board))

    while len(fringe) is not 0:
        min_p_board = heapq.heappop(fringe)[1]
        if min_p_board.tiles == goal_tiles:
            return min_p_board.solution()
        for successor in min_p_board.get_successors():
            # when considering the successors of a search node, don't enqueue a successor if its board is the same as
            # the board of the previous search node. This is to avoid adding search nodes corresponding to the same
            # board on the priority queue too many times. This critical optimization comes from
            # http://coursera.cs.princeton.edu/algs4/assignments/8puzzle.html
            if min_p_board.prev is None or not successor.same_as(min_p_board.prev):
                heapq.heappush(fringe, (successor.priority, successor))


# get the file path from script parameters
initial_tiles = read_file(sys.argv[1])
initial_board = Board(None, initial_tiles, 'Initial', 0)

if is_solvable(initial_board):
    print solve(initial_board)
else:
    print 'No solution possible'

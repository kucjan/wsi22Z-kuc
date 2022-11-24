import sys
from math import inf
from random import choice
import copy
import numpy as np
from heuristics import h_pick

# give an absolute path to submodule: two-player-games
PATH = "/Users/janekkuc/Desktop/PW/Sem7/WSI/wsi22Z-kuc/lab3/two-player-games"
sys.path.append(PATH)

from two_player_games.games.Pick import *

class PlayGame(object):
  """Class that is used to simulate gameplay of given type game with minimax algorithm (with alpha-beta pruning)

  Args:
      game (Game): type of game that should be played
      depth_min (int): depth of player MIN in minimax algorithm
      depth_max (int): depth of player MAX in minimax algorithm
  """
  def __init__(self, game, heuristic, depth_min, depth_max):
    self.game = game
    self.h = heuristic
    self.searched_states_min = 0
    self.searched_states_max = 0
    self.min_selected_numbers = []
    self.max_selected_numbers = []
    self.depth_min = depth_min
    self.depth_max = depth_max
    self.max_turn = True
    
  def minimax_ab(self, state, depth, is_maximizing, alpha=-inf, beta=inf):
    """Recursive method to perform minimax algorithm with alpha-beta pruning

    Args:
        state (State): actual game state
        depth (int): actual searching depth
        is_maximizing (bool): if maximizing player is playing now
        alpha (float, optional): initial alpha value; Defaults to -inf.
        beta (float, optional): initial beta value; Defaults to inf.

    Returns:
        best value and move for given player and state
    """
    if depth == 0 or state.is_finished():
      return (self.h(state, self.max_turn, depth, self.depth_min, self.depth_max), None)
    
    moves = state.get_moves()
    values = []
    
    if is_maximizing:
      value = -inf
      for move in moves:
        self.count_states()
        new_state = copy.deepcopy(state)
        value = max(value, self.minimax_ab(new_state.make_move(move), depth-1, False, alpha, beta)[0])
        alpha = max(alpha, value)
        values.append(alpha)
        if alpha >= beta:
          break
      turn_choice = self.choose_move(values, max(values))
      return (max(values), moves[turn_choice])
    else:
      value = inf
      for move in moves:
        self.count_states()
        new_state = copy.deepcopy(state)
        value = min(value, self.minimax_ab(new_state.make_move(move), depth-1, True, alpha, beta)[0])
        beta = min(beta, value)
        values.append(beta)
        if alpha >= beta:
          break
      turn_choice = self.choose_move(values, min(values))
      return (min(values), moves[turn_choice])
    
  def choose_move(self, moves, value):
    """Method to choose random move if there are few moves with same value"""
    same_moves = [i for i, v in enumerate(moves) if v == value]
    return choice(same_moves)
    
  def count_states(self):
    """Method to count searched states to evaluate quality of algorithm in report"""
    if self.max_turn:
      self.searched_states_max += 1
    else:
      self.searched_states_min += 1
      
  def end_eval(self, state):
    """Method to perform final scoreevaluation"""
    winner = state.get_winner()
    if winner is not None:
      if winner.char == '1':
        return 1000
      else:
        return -1000
    else:
      return 0
    
  def max_min_move(self, depth, is_max):
    """Method to execute single move of player"""
    if depth >= 0:
      value, move = self.minimax_ab(state=self.game.state, depth=depth, is_maximizing=is_max)
      print(value)
      print(f'PLAYER CHOICE: {move.number}')
      self.game.state = self.game.state.make_move(move)
    else:
      moves = self.game.state.get_moves()
      self.game.state = self.game.state.make_move(choice(moves))
      self.count_states()
    self.max_turn = not is_max
    
  def play(self):
    """Method to simulate single game play"""
    turns_counter = 0
    while not self.game.state.is_finished():
      turns_counter += 1
      print(f'----- TURN NUMBER: {turns_counter} -----\n')
      self.max_min_move(self.depth_max, is_max=True)
      print(f'{self.game.state}\n')
      if not self.game.state.is_finished():
        self.max_min_move(self.depth_min, is_max=False)
        print(f'{self.game.state}\n')
    result = self.end_eval(self.game.state)
    print(f'searched min states: {self.searched_states_min}')
    print(f'searched max states: {self.searched_states_max}')
    if result == 1000:
      print(f'MAX (1) with depth={self.depth_max} won vs MIN (2) with depth={self.depth_min}\n')
      return (1, self.searched_states_min, self.searched_states_max)
    elif result == -1000:
      print(f'MIN (2) with depth={self.depth_min} won vs MAX (1) with depth={self.depth_max}\n')
      return (-1, self.searched_states_min, self.searched_states_max)
    else:
      print(f'Draw between: MAX (1) with depth={self.depth_max} and MIN (2) with depth={self.depth_min}\n')
      return (0, self.searched_states_min, self.searched_states_max)

if __name__ == '__main__':
  
  NUM_OF_SIM = 1
  
  DEPTH_MIN = 3
  DEPTH_MAX = 5
  
  save = np.zeros((NUM_OF_SIM, 3))
  
  for i in range(NUM_OF_SIM):
    gameplay = PlayGame(game=Pick(), heuristic=h_pick, depth_min=DEPTH_MIN, depth_max=DEPTH_MAX)
    save[i] = gameplay.play()
  
  print(f'MAX wins count: {(save[:,0] == 1).sum()}')
  print(f'MIN wins count: {(save[:,0] == -1).sum()}')
  print(f'draws count: {(save[:,0] == 0).sum()}')
  print(f'average searched states MIN: {np.mean(save[:,1])}')
  print(f'average searched states MAX: {np.mean(save[:,2])}')

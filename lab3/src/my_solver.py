import sys
from math import inf
from itertools import combinations
from random import randint, choice
import copy
import numpy as np

PATH = "/Users/janekkuc/Desktop/PW/Sem7/WSI/wsi22Z-kuc/lab3/two-player-games"
sys.path.append(PATH)

from two_player_games.games.Pick import *

class PlayPick(object):
  def __init__(self, depth_min, depth_max):
    self.game = Pick()
    self.searched_states_min = 0
    self.searched_states_max = 0
    self.min_selected_numbers = []
    self.max_selected_numbers = []
    self.depth_min = depth_min
    self.depth_max = depth_max
    self.max_turn = True
    
  def minimax_ab(self, state, depth, is_maximizing, alpha=-inf, beta=inf):
    if depth == 0 or state.is_finished():
      return (self.h(state, self.max_turn, depth), None)
    
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
    same_moves = [i for i, v in enumerate(moves) if v == value]
    return choice(same_moves)
    
  def count_states(self):
    """Method to count searched states to evaluate quality of algorithm in report"""
    if self.max_turn:
      self.searched_states_max += 1
    else:
      self.searched_states_min += 1
    
  def h(self, state, is_max, depth):
    if is_max: eval_factor = (self.depth_max-depth)
    else: eval_factor = self.depth_min-depth
    if state.is_finished():
      winner = state.get_winner()
      if winner is not None:
        if winner.char == '1':
          return 1000-eval_factor
        else:
          return -1000+eval_factor
      else:
        return 0
    elif len(state.current_player_numbers) < state.n-1:
      return 0
    else:
      current_player_wins = self.heur_eval(state, True)
      opposite_player_wins = self.heur_eval(state, False)
      if is_max: return current_player_wins - opposite_player_wins
      else: return -(current_player_wins - opposite_player_wins)
      
  def end_eval(self, state):
    winner = state.get_winner()
    if winner is not None:
      if winner.char == '1':
        return 1000
      else:
        return -1000
    else:
      return 0
  
  def heur_eval(self, state, current):
    if current: numbers = state.current_player_numbers
    else: numbers = state.other_player_numbers
    options = list(combinations(numbers, state.n-1))
    wins = set()
    for option in options:
      winning_number = state.aim_value - sum(option)
      if winning_number not in state.selected_numbers and (0 < winning_number <= state.max_number):
        wins.add(winning_number)
    return len(wins)
    
  def max_min_move(self, depth, is_max):
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
    gameplay = PlayPick(depth_min=DEPTH_MIN, depth_max=DEPTH_MAX)
    save[i] = gameplay.play()
  
  print(f'MAX wins count: {(save[:,0] == 1).sum()}')
  print(f'MIN wins count: {(save[:,0] == -1).sum()}')
  print(f'draws count: {(save[:,0] == 0).sum()}')
  print(f'average searched states MIN: {np.mean(save[:,1])}')
  print(f'average searched states MAX: {np.mean(save[:,2])}')

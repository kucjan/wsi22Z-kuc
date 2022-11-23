import sys
from math import inf
from itertools import combinations
from random import randint
from time import sleep
import copy

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
      return (self.h(state), None)
    
    moves = state.get_moves()
    values = []
    
    if is_maximizing:
      value = -inf
      for move in moves:
         # save searched states count to evaluate quality of algorithm in report
        self.searched_states_max += 1
        new_state = copy.deepcopy(state)
        value = max(value, self.minimax_ab(new_state.make_move(move), depth-1, False, alpha, beta)[0])
        alpha = max(alpha, value)
        values.append(alpha)
        if alpha >= beta:
          turn_choice = values.index(alpha)
          return (beta, moves[turn_choice])
      turn_choice = values.index(alpha)
      return (alpha, moves[turn_choice])
    else:
      value = inf
      for move in moves:
        # save searched states count to evaluate quality of algorithm in report
        self.searched_states_min += 1
        new_state = copy.deepcopy(state)
        value = min(value, self.minimax_ab(new_state.make_move(move), depth-1, True, alpha, beta)[0])
        beta = min(beta, value)
        values.append(beta)
        if alpha >= beta:
          turn_choice = values.index(beta)
          return (alpha, moves[turn_choice])
      turn_choice = values.index(beta)
      return (beta, moves[turn_choice])
    
  def h(self, state):
    if state.is_finished():
      winner = state.get_winner()
      if winner is not None:
        if winner.char == '1':
          return 1000
        else:
          return -1000
      else:
        return 0
    elif len(state.current_player_numbers) < state.n-1:
      return 0
    else:
      options = list(combinations(state.current_player_numbers, state.n-1))
      wins = set()
      for option in options:
        winning_number = state.aim_value - sum(option)
        if winning_number not in state.selected_numbers:
          wins.add(winning_number)
      return len(wins)
    
  def max_min_move(self, depth, is_max):
    if depth >= 0:
      _, move = self.minimax_ab(state=self.game.state, depth=depth, is_maximizing=is_max)
      self.game.state = self.game.state.make_move(move)
    else:
      moves = self.game.state.get_moves()
      rand_move_ind = randint(0, len(moves)-1)
      self.game.state = self.game.state.make_move(rand_move_ind)
    self.max_turn = not is_max
    
  def play(self):
    print(self.game.state)
    while not self.game.state.is_finished():
      self.max_min_move(self.depth_max, is_max=True)
      print(self.game.state)
      if not self.game.state.is_finished():
        self.max_min_move(self.depth_min, is_max=False)
        print(self.game.state)
    result = self.h(self.game.state)
    print(f'searched min states: {self.searched_states_min}')
    print(f'searched max states: {self.searched_states_max}')
    if result == 1000:
      print(f'MAX with depth={self.depth_max} won vs MIN with depth={self.depth_min}')
      return (1, self.searched_states_max, self.searched_states_min)
    elif result == -1000:
      print(f'MIN with depth={self.depth_min} won vs MAX with depth={self.depth_max}')
      return (-1, self.searched_states_max, self.searched_states_min)
    else:
      print(f'Draw between: MAX with depth={self.depth_max} and MIN with depth={self.depth_min}')
      return (0, self.searched_states_max, self.searched_states_min)

if __name__ == '__main__':
  gameplay = PlayPick(depth_min=5, depth_max=7)
  gameplay.play()

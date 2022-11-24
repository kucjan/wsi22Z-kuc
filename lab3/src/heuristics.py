from itertools import combinations

def h_pick(state, is_max, depth, depth_min, depth_max):
  """Method to perform heuristic score for nodes above maximum depth"""
  if is_max: eval_factor = (depth_max-depth)
  else: eval_factor = depth_min-depth
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
    current_player_wins = heur_eval_pick(state, True)
    opposite_player_wins = heur_eval_pick(state, False)
    if is_max: return current_player_wins - opposite_player_wins
    else: return -(current_player_wins - opposite_player_wins)

def heur_eval_pick(state, current):
  if current: numbers = state.current_player_numbers
  else: numbers = state.other_player_numbers
  options = list(combinations(numbers, state.n-1))
  wins = set()
  for option in options:
    winning_number = state.aim_value - sum(option)
    if winning_number not in state.selected_numbers and (0 < winning_number <= state.max_number):
      wins.add(winning_number)
  return len(wins)
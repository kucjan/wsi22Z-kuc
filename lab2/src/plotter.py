import matplotlib.pyplot as plt

def plot_alg_run(best_vals, max_vals, min_vals, avg_vals):
  steps = len(best_vals)
  plt.figure()
  plt.plot(range(steps), best_vals, 'k', label='best values')
  plt.plot(range(steps), max_vals, 'g', label='max values')
  plt.plot(range(steps), min_vals, 'r', label='min values')
  plt.plot(range(steps), avg_vals, 'b', label='average values')
  plt.legend()
  plt.xlabel('step number')
  plt.ylabel('obj fun values')
  plt.show()
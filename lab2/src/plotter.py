import matplotlib.pyplot as plt
import numpy as np

def plot_alg_run(best_vals, max_vals, min_vals, avg_vals, pc, pm, pop_size):
  save_dir = '/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab2/'
  steps = len(best_vals)
  plt.figure()
  plt.plot(range(steps), best_vals, 'k', label='best values')
  plt.legend(loc='upper right')
  plt.xlabel('steps')
  plt.ylabel('obj fun values')
  plt.savefig(save_dir + f'bestvals_pc{pc}_pm{pm}_s{pop_size}.pdf')
  
  plt.figure()
  plt.plot(range(steps), max_vals, 'g', label='max values')
  plt.legend(loc='upper right')
  plt.xlabel('steps')
  plt.ylabel('obj fun values')
  plt.savefig(save_dir + f'maxvals_pc{pc}_pm{pm}_s{pop_size}.pdf')
  
  plt.figure()
  plt.plot(range(steps), min_vals, 'r', label='min values')
  plt.legend(loc='upper right')
  plt.xlabel('steps')
  plt.ylabel('obj fun values')
  plt.savefig(save_dir + f'minvals_pc{pc}_pm{pm}_s{pop_size}.pdf')
  
  plt.figure()
  plt.plot(range(steps), avg_vals, 'b', label='average values')
  plt.legend(loc='upper right')
  plt.xlabel('steps')
  plt.ylabel('obj fun values')
  plt.savefig(save_dir + f'avgvals_pc{pc}_pm{pm}_s{pop_size}.pdf')
  
  plt.show()
  
def group_test_plot(best_values, max_values, solution_values, pc, pm, pop_size):
  save_dir = '/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab2/'
  steps = range(0, best_values.shape[1])
  
  max_best_values = []
  avg_best_values = []
  min_best_values = []
  max_max_values = []
  min_max_values = []
  avg_max_values = []
      
  for it in steps:
    max_best_values.append(max(best_values[:, it]))
    avg_best_values.append(sum(best_values[:, it])/len(best_values[:, it]))
    min_best_values.append(min(best_values[:, it]))
    max_max_values.append(max(max_values[:, it]))
    min_max_values.append(min(max_values[:, it]))
    avg_max_values.append(sum(max_values[:, it])/len(max_values[:, it]))
    
  
  plt.figure()
  # plt.plot(steps, avg_max_values, 'b', label='max values')
  plt.fill_between(steps, min_best_values, max_best_values, color='b', alpha=0.2, label='min to max')
  # plt.plot(steps, max_best_values, 'g', label='best values')
  plt.plot(steps, avg_best_values, 'r', label='avg best values')
  plt.legend(loc='lower left')
  plt.xlabel('steps')
  plt.ylabel('obj fun values')
  plt.savefig(save_dir + f'groupplot_pc{pc}_pm{pm}_s{pop_size}.pdf')
  plt.show()
  
  result = {
    'cross_prob': pc,
    'best_value': max(solution_values),
    'mean_best_values': np.mean(solution_values),
    'std_solution_values': np.std(solution_values)
  }
  
  print(result)
  
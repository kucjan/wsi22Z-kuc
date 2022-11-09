from my_solver import MySolver
from rocket_landing import land_rocket, generate_population
from plotter import group_test_plot
import numpy as np
import json

NUM_OF_RUNS = 25

CHROM_SIZE = 200
POPULATION_SIZE = 500

PC = 0.8
PM = 0.1
MAX_ITERATIONS = 1000

if __name__ == '__main__':
  
  solution_values = []
  best_values = np.array([])
  max_values = np.array([])
  
  for i in range(NUM_OF_RUNS):
    
    print(f'iteration: {i}')
    
    pop0 = generate_population(chrom_size=CHROM_SIZE, pop_size=POPULATION_SIZE)
    
    solver1 = MySolver(pc=PC, pm=PM, max_it=MAX_ITERATIONS)
    
    solution, best_vals_i, max_vals_i = solver1.solve(problem=land_rocket, pop0=pop0, show_plots=False)
    
    solution_values.append(solution['best_value'])
    
    json_object = json.dumps(solution, indent=len(solution))
    
    save_dir = '/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab2/'
    
    with open(save_dir+'params_pc{}_pm{}_s{}_run_{}.json'.format(PC, PM, POPULATION_SIZE, i), 'w') as outfile:
      outfile.write(json_object)
    
    if i == 0:
      best_values = np.array(best_vals_i)
      max_values = np.array(max_vals_i)
    else:
      best_values = np.vstack([best_values, best_vals_i])
      max_values = np.vstack([max_values, max_vals_i])
      
  group_test_plot(best_values, max_values, solution_values, PC, PM, POPULATION_SIZE)

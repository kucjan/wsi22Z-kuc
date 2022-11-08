from solver import Solver
from rocket_landing import land_rocket, generate_population
import numpy as np
from random import uniform, randint
from plotter import plot_alg_run

class MySolver(Solver):
  
  def __init__(self, pc, pm, max_it):
    self.pc = pc  # probability of crossing
    self.pm = pm  # probability of mutation
    self.max_it = max_it
    self.best_vals = []
    self.max_vals = []
    self.min_vals = []
    self.avg_vals = []
    
  def get_parameters(self, problem_size, max_repeat, best_value, best_chrom, iterations):
    params = {
      "cross_probability": self.pc,
      "mutation_probability": self.pm,
      "max_iterations": self.max_it,
      "max_repeats": max_repeat,
      "problem_size": problem_size,
      "best_value": best_value,
      "best_chrom": best_chrom,
      "num_of_iter": iterations
    }
    
    return params
  
  def solve(self, problem, pop0):
    """A method to execute main loop of genetic algorithm"""
    
    MAX_REPEAT = 300 # max number of iterations with same solution
    MAX_EVALS = 100000 # max number of evaluations of objective function (fixed budget)
    
    f_vals, best_chrom, best_val = self.rate_population(problem, pop0)
    generation = pop0
    i = 0
    rep = 0
    print('i: {}  best_val: {}  min_val: {}  max_val: {}  mean_val: {}'
      .format(i, best_val, min(f_vals), max(f_vals), sum(f_vals)/len(f_vals)))
    while i < self.max_it and rep < MAX_REPEAT and problem.called < MAX_EVALS:
      selection = self.roulette_selection(f_vals, generation)
      generation = self.cross_and_mutate(selection, self.pc, self.pm)
      print(problem.called)
      f_vals, new_chrom, new_val = self.rate_population(problem, generation)
      if new_val > best_val:
        best_chrom = new_chrom
        best_val = new_val
        rep = 0
      else:
        rep += 1
      self.best_vals.append(best_val)
      # self.max_vals.append((max(f_vals), np.argmax(f_vals)))
      # self.min_vals.append((min(f_vals), np.argmin(f_vals)))
      self.max_vals.append(max(f_vals))
      self.min_vals.append(min(f_vals))
      self.avg_vals.append(sum(f_vals)/len(f_vals))
      i += 1
      print('i: {}  best_val: {}  min_val: {}  max_val: {}  mean_val: {}'
            .format(i, best_val, min(f_vals), max(f_vals), sum(f_vals)/len(f_vals)))
      
    plot_alg_run(self.best_vals, self.max_vals, self.min_vals, self.avg_vals)
        
    return self.get_parameters(len(pop0), MAX_REPEAT, best_val, best_chrom, i)

  def rate_population(self, problem, population):
    """A method that calculates objective function values for all chromosoms of generation"""

    best_chrom = population[0]; best_val = problem(population[0])
    f_vals = [best_val]
    for chrom in population[1:]:
      chrom_val = problem(chrom)
      f_vals.append(chrom_val)
      if chrom_val > best_val:
        best_chrom = chrom
        best_val = chrom_val
    return f_vals, best_chrom, best_val
  
  def roulette_selection(self, values, population):
    """A method to execute roullete selection of population"""
    
    probs = []; sum_values = 0; next_gen = []
    cons = min(values) if min(values) < 0 else 0
    sum_values = sum(values) + cons*len(values)
    
    prev_prob = 0
    for v in values:
      new_prob = prev_prob + (v+cons)/sum_values
      probs.append((prev_prob, new_prob))
      prev_prob = new_prob
    
    while len(next_gen) < len(population):
      rand = uniform(0.0, 1.0)
      for i in range(len(population)):
        if probs[i][0] < rand <= probs[i][1]:
          next_gen.append(population[i])
        
    return next_gen
  
  def cross_and_mutate(self, gen, pc, pm):
    """A method to execute crossing and mutation of chromosomes"""
    chrom_size = len(gen[0])
    it = iter(range(len(gen)))
    for i, j in zip(it, it):
      if uniform(0.0, 1.0) <= pc:
        cut = randint(0, chrom_size-1)
        new_chrom1 = gen[i][0:cut]; new_chrom2 = gen[j][0:cut]
        new_chrom1.extend(gen[j][cut:]); new_chrom2.extend(gen[i][cut:])
        gen[i] = new_chrom1
        gen[j] = new_chrom2
      self.mutate(gen[i], pm)
      self.mutate(gen[j], pm) 
      
    return gen
  
  def mutate(self, chrom, pm):
    """A method to execute mutation on single chromosome"""
    for i in range(len(chrom)):
      if uniform(0.0, 1.0) <= pm:
        chrom[i] = 1 - chrom[i]
  
if __name__ == '__main__':
  
  pop0 = generate_population(chrom_size=200, pop_size=500)
  
  solver1 = MySolver(pc=0.85, pm=0.15, max_it=2000)
  
  solution = solver1.solve(problem=land_rocket, pop0=pop0)
  
  print(solution)
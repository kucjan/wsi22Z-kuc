from solver import Solver
from math import exp
import numpy as np
from plotter import generate_plots

class MySolver(Solver):

    def __init__(self, e, beta, eps, max_it):
      self.e = e  # initial step length
      self.beta = beta  # step change factor
      self.eps = eps  # tolerance
      self.max_it = max_it
      self.x_list = []
      self.f_list = []
      self.g_list = []

    def get_parameters(self, x0, init_e, it, stop_crit, xi, fi):
        
      params = {
        "x0": x0,
        "initial_e": init_e,
        "e": self.e,
        "beta": self.beta,
        "eps": self.eps,
        "max_it": self.max_it,
        "end_it": it,
        "stop_crit": stop_crit,
        "min_xi": xi,
        "f(min_xi)": fi
      }
      
      return params

    def solve(self, problem, x0):
      
      # get function and derivative from defined problem
      f, df = problem
      
      # find f and df for x0
      f0 = f(x0)
      g0 = df(x0)

      srch_dir = -g0   # search direction
      
      # initial step and new point cords
      xi = x0 + self.e*srch_dir
    
      # fin f and df for new point
      fi = f(xi)
      gi = df(xi)
      
      # initialize lists for saving x, f and df values
      # append first 2 elements
      self.x_list = np.array([x0]); self.x_list = np.vstack([self.x_list, xi])
      self.f_list = np.array([f0]); self.f_list = np.vstack([self.f_list, fi])
      self.g_list = np.array([g0]); self.g_list = np.vstack([self.g_list, gi])
      
      # save previous points cords and previous f value
      prev_x = x0
      prev_f = f0
      
      # save initial e parameter value
      init_e = self.e 
      
      # initial algorithm stop criterium
      stop_crit = 'max iterations'
      
      for i in range(2, self.max_it+1):
        if np.all(abs(gi) > self.eps) and np.linalg.norm(xi - prev_x) > self.eps:
          print(xi, i, fi, gi)
          if fi < prev_f:
            prev_x = self.x_list[i-1]
            srch_dir = -gi
          else:
            xi -= srch_dir
            prev_x = self.x_list[i-2]
            fi = f(xi)
            self.e *= self.beta
          xi, fi, gi, prev_f = self.step(xi, srch_dir, fi, f, df)
        else:
          if np.all(abs(gi) <= self.eps):
            stop_crit = "gradient close to zero"
          else:
            stop_crit = "no improvement"
          break
      
      generate_plots(self.x_list, self.f_list, self.e, type(x0) != np.ndarray, f)
      
      return self.get_parameters(x0, init_e, i, stop_crit, xi, fi)
         
    def step(self, xi, srch_dir, fi, f, df):
      """A method that executes one step of algorithm and saves historical data"""
      xi += self.e*srch_dir
      prev_f = fi
      fi = f(xi)
      gi = df(xi)
      self.x_list = np.vstack([self.x_list, xi])
      self.f_list = np.vstack([self.f_list, fi])
      self.g_list = np.vstack([self.g_list, gi])
      return xi, fi, gi, prev_f
    
if __name__ == '__main__':
  
  problem1 = [lambda x: pow(x,4)/4, lambda x: pow(x, 3)]
  problem2 = [lambda x: 2 - exp(-pow(x[0],2) - pow(x[1],2)) - exp(-pow(x[0]+1.5,2) - pow(x[1]-2,2))/2,
              lambda x: np.array([
                2*x[0] * exp(-pow(x[0],2) - pow(x[1],2)) + (x[0] + 1.5) * exp(-pow(x[0]+1.5,2) - pow(x[1]-2,2)),
                2*x[1] * exp(-pow(x[0],2) - pow(x[1],2)) + (x[1] - 2) * exp(-pow(x[0]+1.5,2) - pow(x[1]-2,2))
              ])]
  
  solver1d = MySolver(e=0.9, beta=0.9, eps=0.00001, max_it=10000)
  solution = solver1d.solve(problem1, -1.5)
  
  # solver2d = MySolver(e=0.001, beta=0.9, eps=0.000001, max_it=10000)
  # solution = solver2d.solve(problem2, np.array([0.2,0.6]))
  
  print(solution)
  
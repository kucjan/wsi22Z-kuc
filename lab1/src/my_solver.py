from solver import Solver
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
            xi = self.x_list[i-1]
            prev_x = self.x_list[i-2]
            fi = f(xi)
            self.e -= self.beta
          xi, fi, gi, prev_f = self.step(xi, srch_dir, fi, f, df)
        else:
          if np.all(abs(gi) <= self.eps):
            stop_crit = "gradient close to zero"
          else:
            stop_crit = "no improvement"
          break

      print(f'MIN x: {xi}')
      print(f'F(min_x): {fi}')
      
      self.generate_plots(type(x0) != np.ndarray, f)
      
      return self.get_parameters(x0, init_e, i, stop_crit, xi, fi)
         
    def step(self, xi, srch_dir, fi, f, df):
      xi += self.e*srch_dir
      prev_f = fi
      fi = f(xi)
      gi = df(xi)
      self.x_list = np.vstack([self.x_list, xi])
      self.f_list = np.vstack([self.f_list, fi])
      self.g_list = np.vstack([self.g_list, gi])
      return xi, fi, gi, prev_f
    
    def generate_plots(self, is_1d, f):
      if is_1d:
        fig1 = plt.figure()
        plt.scatter(self.x_list[0], self.f_list[0], marker='x', color="c", s=10, label='start point')
        plt.scatter(self.x_list[1:-2], self.f_list[1:-2], marker='o', color="k", s=1, label='alg. path')
        plt.scatter(self.x_list[-1], self.f_list[-1], marker='x', color="r", s=10, label='end point')
        iters = np.linspace(self.x_list[0], self.x_list[-1], 101)
        f_iters = []
        for it in iters:
          f_iters.append(f(it))
        plt.plot(iters, f_iters, color="b", alpha=0.3, label='f(x)')
        plt.xlabel('xi')
        plt.ylabel('F(xi)')
        plt.legend()
        fig1_name = f'traj_{self.x_list[0]}_{self.e}.pdf'
        fig1.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig1_name}')
        
        fig2 = plt.figure()
        plt.scatter(0, self.f_list[0], marker='x', color='cyan', s=10, label='f(start point)')
        plt.scatter(range(1,len(self.f_list)-1), self.f_list[1:-1], marker='o', color='black', s=1, label='f(xi) path')
        plt.scatter(len(self.x_list), self.f_list[-1], marker='x', color='red', s=10, label='f(end point)')
        plt.xlabel('iteration')
        plt.ylabel('F(xi)')
        plt.legend()
        plt.yscale('log')
        fig2_name = f'fcel_log_{self.x_list[0]}_{self.e}.pdf'
        fig2.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig2_name}')
        plt.show()
      else:
        fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
        x = np.linspace(-5, 5, 101)
        y = np.linspace(-5, 5, 101)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(len(x)):
          for j in range(len(y)):
            Z[i,j] = f([x[i],y[j]])
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)
        ax.scatter(self.x_list[0,0], self.x_list[0,1], self.f_list[0], color='cyan', s=6, label='start point')
        ax.scatter(self.x_list[1:-2,0], self.x_list[1:-2,1], self.f_list[1:-2], color='black', s=2, label='alg. path')
        ax.scatter(self.x_list[-1,0], self.x_list[-1,1], self.f_list[-1], color='red', s=6, label='end point')
        ax.legend(loc='upper left')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('F(x1,x2)')
        ax.set_zlim(0.9*np.min(Z), 1.1*np.max(Z))
        fig1.colorbar(surf, shrink=0.5, aspect=5)
        fig1_name = f'surf_{self.x_list[0]}_{self.e}.pdf'
        fig1.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig1_name}')
        
        fig2 = plt.figure()
        plt.contour(X, Y, Z, alpha=0.6)
        plt.colorbar()
        plt.scatter(self.x_list[0,0], self.x_list[0,1], color='cyan', s=5, label='start point')
        plt.scatter(self.x_list[1:-2,0], self.x_list[1:-2,1], color='black', s=3, label='alg. path')
        plt.scatter(self.x_list[-1,0], self.x_list[-1,1], color='red', s=5, label='end point')
        plt.legend(loc='upper left')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(np.min(self.x_list[:,0])-2, np.max(self.x_list[:,0])+2)
        plt.ylim(np.min(self.x_list[:,1])-2, np.max(self.x_list[:,1])+2)
        fig2_name = f'contour_{self.x_list[0]}_{self.e}.pdf'
        fig2.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig2_name}')
        plt.show()
    
if __name__ == '__main__':
  
  problem1 = [lambda x: pow(x,4)/4, lambda x: pow(x, 3)]
  problem2 = [lambda x: 2 - exp(-pow(x[0],2) - pow(x[1],2)) - exp(-pow(x[0]+1.5,2) - pow(x[1]-2,2))/2,
              lambda x: np.array([
                2*x[0] * exp(-pow(x[0],2) - pow(x[1],2)) + (x[0] + 1.5) * exp(-pow(x[0]+1.5,2) - pow(x[1]-2,2)),
                2*x[1] * exp(-pow(x[0],2) - pow(x[1],2)) + (x[1] - 2) * exp(-pow(x[0]+1.5,2) - pow(x[1]-2,2))
              ])]
  
  solver1d = MySolver(e=1, beta=0.01, eps=0.00001, max_it=10000)
  
  solver2d = MySolver(e=0.2, beta=0.001, eps=0.000001, max_it=1000)
  
  solution = solver1d.solve(problem1, -1.5)
  # solution = solver2d.solve(problem2, np.array([2,-2]))
  
  print(solution)
  
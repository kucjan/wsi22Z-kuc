import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_plots(x_list, f_list, e, is_1d, f):
  """A method that generates, displays and saves proper plots for exact problems"""
  if is_1d:
    fig1 = plt.figure()
    plt.scatter(x_list[0], f_list[0], marker='x', color="c", s=10, label='start point')
    plt.scatter(x_list[1:-2], f_list[1:-2], marker='o', color="k", s=1, label='alg. path')
    plt.scatter(x_list[-1], f_list[-1], marker='x', color="r", s=10, label='end point')
    iters = np.linspace(x_list[0], x_list[-1], 101)
    f_iters = []
    for it in iters:
      f_iters.append(f(it))
    plt.plot(iters, f_iters, color="b", alpha=0.3, label='f(x)')
    plt.xlabel('xi')
    plt.ylabel('F(xi)')
    plt.legend()
    fig1_name = f'traj_{x_list[0]}_{e}.pdf'
    
    fig2 = plt.figure()
    plt.scatter(0, f_list[0], marker='x', color='cyan', s=10, label='f(start point)')
    plt.scatter(range(1,len(f_list)-1), f_list[1:-1], marker='o', color='black', s=1, label='f(xi) path')
    plt.scatter(len(x_list), f_list[-1], marker='x', color='red', s=10, label='f(end point)')
    plt.xlabel('iteration')
    plt.ylabel('F(xi)')
    plt.legend()
    plt.yscale('log')
    fig2_name = f'fcel_log_{x_list[0]}_{e}.pdf'
    plt.show()
    # fig1.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig1_name}')
    # fig2.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig2_name}')
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
    ax.scatter(x_list[0,0], x_list[0,1], f_list[0], color='cyan', s=6, label='start point')
    ax.scatter(x_list[1:-2,0], x_list[1:-2,1], f_list[1:-2], color='black', s=2, label='alg. path')
    ax.scatter(x_list[-1,0], x_list[-1,1], f_list[-1], color='red', s=6, label='end point')
    ax.legend(loc='upper left')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('F(x1,x2)')
    ax.set_zlim(0.9*np.min(Z), 1.1*np.max(Z))
    fig1.colorbar(surf, shrink=0.5, aspect=5)
    fig1_name = f'surf_{x_list[0]}_{e}.pdf'
    
    fig2 = plt.figure()
    plt.contour(X, Y, Z, alpha=0.6)
    plt.colorbar()
    plt.scatter(x_list[0,0], x_list[0,1], color='cyan', s=5, label='start point')
    plt.scatter(x_list[1:-2,0], x_list[1:-2,1], color='black', s=3, label='alg. path')
    plt.scatter(x_list[-1,0], x_list[-1,1], color='red', s=5, label='end point')
    plt.legend(loc='upper left')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(np.min(x_list[:,0])-2, np.max(x_list[:,0])+2)
    plt.ylim(np.min(x_list[:,1])-2, np.max(x_list[:,1])+2)
    fig2_name = f'contour_{x_list[0]}_{e}.pdf'
    plt.show()
    # fig1.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig1_name}')
    # fig2.savefig(f'/Users/janekkuc/Desktop/PW/Sem7/WSI/wykresy_lab1/{fig2_name}')
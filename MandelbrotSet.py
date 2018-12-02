import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import theano
import theano.tensor as T

try:
    os.mkdir(__file__.split('.')[0])
except OSError:
    pass

C = T.cmatrix()
M = T.cmatrix()
d = T.scalar()

def step(X):
    return T.pow(X, d) + C

result, _ = theano.scan(fn=step,
                        outputs_info=M,
                        n_steps=50)

f = theano.function([M,C,d], result, allow_input_downcast=True)

def plot_mandelbrot(K    = 4000,
                    save = True):
    x,y = np.meshgrid(np.linspace(-2.0, 1.0, K), np.linspace(-1.5, 1.5, K))
    Z_n = f(np.zeros((K,K)), x+y*1j, 2.0)
    mandelbrot = np.sum(np.absolute(Z_n) < 2.0, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    ax.contourf(x, y, mandelbrot, cmap='gray')
    if save:
        plt.savefig(os.path.join(__file__.split('.')[0], __file__.split('.')[0]+'.png'), dpi=600)
    plt.show()


def plot_multibrot(K    = 4000,
                   save = True):
    x,y = np.meshgrid(np.linspace(-2.0, 1.0, K), np.linspace(-1.5, 1.5, K))
    Z_n = f(np.zeros((K,K)), x+y*1j, 0.0)
    mandelbrot = np.sum(np.absolute(Z_n) < 2.0, axis=0)

    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta/2, f[-1] + delta/2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    mplot = ax.imshow(mandelbrot,
                      extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
                      origin='lower',
                      cmap='gray')

    def update(d):
        label = 'timestep {0}'.format(d)
        print label

        Z_n = f(np.zeros((K,K)), x+y*1j, d)
        mandelbrot = np.sum(np.absolute(Z_n) < 2.0, axis=0)
        mplot.set_data(mandelbrot)

    anim = FuncAnimation(fig, update, frames=np.arange(0.0, 10.0, 0.05), interval=50)

    if save:
        anim.save(os.path.join(__file__.split('.')[0], 'Multibrot.gif'), dpi=200, writer='imagemagick')
    plt.show()


def plot_mandelbar(K    = 4000,
                   save = True):
    def step(X):
        return T.square(T.conj(X)) + C

    result, _ = theano.scan(fn=step,
                            outputs_info=M,
                            n_steps=50)

    f = theano.function([M,C], result, allow_input_downcast=True)

    x,y = np.meshgrid(np.linspace(-2.0, 1.0, K), np.linspace(-1.5, 1.5, K))
    Z_n = f(np.zeros((K,K)), x+y*1j)
    mandelbar = np.sum(np.absolute(Z_n) < 2.0, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    ax.contourf(x, y, mandelbar, cmap='gray')
    if save:
        plt.savefig(os.path.join(__file__.split('.')[0], 'Mandelbar.png'), dpi=600)
    plt.show()


def main():
    plot_mandelbrot(800, False)
    #plot_multibrot(800, True)
    plot_mandelbar(4000, True)

if __name__=='__main__':
    main()

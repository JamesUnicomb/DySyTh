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


def mandelbrot_functions(function = ['mandelbrot', 'mandelbar', 'multibrot'][0],
                         N = 50):
    C = T.cmatrix()
    M = T.cmatrix()
    d = T.scalar()

    if function == 'mandelbrot':
        def step(X):
            return T.square(X) + C
    elif function == 'multibrot':
        def step(X):
            return T.pow(X, d) + C
    elif function == 'mandelbar':
        def step(X):
            return T.square(T.conj(X)) + C

    result, _ = theano.scan(fn=step,
                            outputs_info=M,
                            n_steps=N)

    if (function == 'mandelbrot')or(function == 'mandelbar'):
        f = theano.function([M,C], result, allow_input_downcast=True)
    else:
        f = theano.function([M,C,d], result, allow_input_downcast=True)

    return f


def plot_mandelbrot(K    = 4000,
                    save = True):
    f = mandelbrot_functions(function = 'mandelbrot')

    x,y = np.meshgrid(np.linspace(-2.0, 1.0, K), np.linspace(-1.5, 1.5, K))
    Z_n = f(np.zeros((K,K)), x+y*1j)
    mandelbrot = np.sum(np.absolute(Z_n) < 2.0, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    ax.contourf(x, y, mandelbrot, cmap='Blues_r')
    if save:
        plt.savefig(os.path.join(__file__.split('.')[0], __file__.split('.')[0]+'.png'), dpi=600)
    plt.show()


def plot_multibrot(K    = 4000,
                   save = True):
    f = mandelbrot_functions(function = 'multibrot')

    x,y = np.meshgrid(np.linspace(-2.0, 1.0, K), np.linspace(-1.5, 1.5, K))
    Z_n = f(np.zeros((K,K)), x+y*1j, 0.0)
    mandelbrot = np.sum(np.absolute(Z_n) < 2.0, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    mplot = ax.imshow(mandelbrot,
                      extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
                      origin='lower',
                      cmap='Blues_r')

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
    f = mandelbrot_functions(function = 'mandelbar')

    x,y = np.meshgrid(np.linspace(-2.0, 1.0, K), np.linspace(-1.5, 1.5, K))
    Z_n = f(np.zeros((K,K)), x+y*1j)
    mandelbar = np.sum(np.absolute(Z_n) < 2.0, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_aspect('equal')

    ax.contourf(x, y, mandelbar, cmap='Blues_r')
    if save:
        plt.savefig(os.path.join(__file__.split('.')[0], 'Mandelbar.png'), dpi=600)
    plt.show()


# def mandelbulb(K = 200):
#


def main():
    plot_mandelbrot(4000, True)
    plot_multibrot(800, True)
    plot_mandelbar(4000, True)

if __name__=='__main__':
    main()

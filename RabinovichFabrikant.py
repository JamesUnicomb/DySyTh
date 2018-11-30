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


gamma = T.fscalar()
alpha = T.fscalar()

h = T.fscalar()
N = T.iscalar()
x = T.fvector()

def f(X):
    X_ = T.zeros_like(X)
    X_ = T.set_subtensor(X_[0], X[1] * (X[2] - 1.0 + X[0] * X[0]) + gamma * X[0])
    X_ = T.set_subtensor(X_[1], X[0] * (3.0 * X[2] + 1.0 - X[0] * X[0]) + gamma * X[1])
    X_ = T.set_subtensor(X_[2], -2.0 * X[2] * (alpha + X[0] * X[1]))
    return X_


def step(X):
    k1 = h * f(X)
    k2 = h * f(X + 0.5 * k1)
    k3 = h * f(X + 0.5 * k2)
    k4 = h * f(X + k3)

    X_ = X + (1.0 / 6.0) * k1 + (1.0 / 3.0) * k2 + (1.0 / 3.0) * k3 + (1.0 / 6.0) * k4

    return X_

result, _ = theano.scan(fn=step,
                        outputs_info=x,
                        n_steps=N)

RK4 = theano.function([x,h,gamma,alpha,N],
                      result,
                      allow_input_downcast=True)





def plot_path():
    state_array_a = RK4(np.array([-1.0, 0.0, 0.5]),
                        0.005, 0.87, 1.1, 20000)
    state_array_b = RK4(np.array([-1.0, 0.0, 0.5]),
                        0.02, 0.1, 0.14, 20000)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    fig.set_tight_layout(True)
    ax1.set_axis_off()
    ax1.set_title(r'$\gamma=0.87$, $\alpha=1.1$')
    ax2.set_axis_off()
    ax2.set_title(r'$\gamma=0.1$, $\alpha=0.14$')


    ax1.plot(state_array_a[:,0], state_array_a[:,1], state_array_a[:,2], c='C0', linewidth=0.125)
    ax2.plot(state_array_b[:,0], state_array_b[:,1], state_array_b[:,2], c='C0', linewidth=0.125)
    k=1
    line_a, = ax1.plot(state_array_a[:k,0], state_array_a[:k,1], state_array_a[:k,2], color='C0')
    line_b, = ax2.plot(state_array_b[:k,0], state_array_b[:k,1], state_array_b[:k,2], color='C0')

    def update(k):
        ax1.view_init(30.0, -45.0 + k)
        ax2.view_init(30.0, -45.0 + k)
        label = 'timestep {0}'.format(k)
        print label

        idt = int((1.0 * k) / 360 * len(state_array_a))
        line_a.set_xdata(state_array_a[:idt,0])
        line_a.set_ydata(state_array_a[:idt,1])
        line_a.set_3d_properties(state_array_a[:idt,2])

        line_b.set_xdata(state_array_b[:idt,0])
        line_b.set_ydata(state_array_b[:idt,1])
        line_b.set_3d_properties(state_array_b[:idt,2])

        return ax1, ax2

    anim = FuncAnimation(fig, update, frames=np.arange(0, 360), interval=50)
    anim.save(os.path.join(__file__.split('.')[0], __file__.split('.')[0]+'.gif'), dpi=80, writer='imagemagick')
    plt.show()


def main():
    plot_path()

if __name__=='__main__':
    main()

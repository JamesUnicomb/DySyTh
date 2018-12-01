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

mass     = T.fscalar()
length   = T.fscalar()
gravity  = T.fscalar()

h = T.fscalar()
N = T.iscalar()
x = T.fvector()

def f(X):
    X_ = T.zeros_like(X)
    X_ = T.set_subtensor(X_[0], (6.0 / (mass * length * length)) * \
                                ((2.0 * X[2] - 3.0 * T.cos(X[0] - X[1]) * X[3]) / \
                                (16.0 - 9.0 * T.square(T.cos(X[0] - X[1])))))
    X_ = T.set_subtensor(X_[1], (6.0 / (mass * length * length)) * \
                                (8.0 * X[3] - 3.0 * T.cos(X[0] - X[1]) * X[2]) / \
                                (16.0 - 9.0 * T.square(T.cos(X[0] - X[1]))))
    X_ = T.set_subtensor(X_[2], -0.5 * mass * length * length * (X_[0] * X_[1] * T.sin(X[0] - X[1]) + \
                                                                 3.0 * gravity / length * T.sin(X[0])))
    X_ = T.set_subtensor(X_[3], -0.5 * mass * length * length * (-X_[0] * X_[1] * T.sin(X[0] - X[1]) + \
                                                                 gravity / length * T.sin(X[1])))
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

RK4 = theano.function([x,h,mass,length,gravity,N],
                      result,
                      allow_input_downcast=True)




def plot_path():
    theta1 = np.random.uniform(0.0, 2.0 * np.pi)
    theta2 = np.random.uniform(0.0, 2.0 * np.pi)
    l      = 1.0
    m      = 1.0
    g      = 9.81

    state_array = RK4(np.array([theta1, theta2, 0.0, 0.0]),
                         0.005, m, l, g, 4000)

    x1 = l * np.cos(state_array[:,0] - np.pi / 2)
    y1 = l * np.sin(state_array[:,0] - np.pi / 2)

    x2 = x1 + l * np.cos(state_array[:,1] - np.pi / 2)
    y2 = y1 + l * np.sin(state_array[:,1] - np.pi / 2)



    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)
    ax.set_axis_off()
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.0, 3.0])
    ax.set_aspect('equal')

    k=0
    pendulum_line, = ax.plot([0.0, x1[k], x2[k]], [0.0, y1[k], y2[k]], 'k')
    path_line,     = ax.plot(x1[:k], y1[:k], c='C0', linewidth=0.125)

    def update(k):
        label = 'timestep {0}'.format(k)
        print label

        idt = int((1.0 * k) / 800 * len(state_array))
        pendulum_line.set_xdata([0.0, x1[idt], x2[idt]])
        pendulum_line.set_ydata([0.0, y1[idt], y2[idt]])

        path_line.set_xdata(x2[:idt])
        path_line.set_ydata(y2[:idt])

        return ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, 800), interval=50)
    anim.save(os.path.join(__file__.split('.')[0], __file__.split('.')[0]+'.gif'), dpi=80, writer='imagemagick')
    plt.show()



def plot_zero_crossing(K = 600):
    x = T.ftensor3()

    def f(X):
        X_ = T.zeros_like(X)
        X_ = T.set_subtensor(X_[:,:,0], (6.0 / (mass * length * length)) * \
                                        ((2.0 * X[:,:,2] - 3.0 * T.cos(X[:,:,0] - X[:,:,1]) * X[:,:,3]) / \
                                        (16.0 - 9.0 * T.square(T.cos(X[:,:,0] - X[:,:,1])))))
        X_ = T.set_subtensor(X_[:,:,1], (6.0 / (mass * length * length)) * \
                                        (8.0 * X[:,:,3] - 3.0 * T.cos(X[:,:,0] - X[:,:,1]) * X[:,:,2]) / \
                                        (16.0 - 9.0 * T.square(T.cos(X[:,:,0] - X[:,:,1]))))
        X_ = T.set_subtensor(X_[:,:,2], -0.5 * mass * length * length * (X_[:,:,0] * X_[:,:,1] * T.sin(X[:,:,0] - X[:,:,1]) + \
                                                                     3.0 * gravity / length * T.sin(X[:,:,0])))
        X_ = T.set_subtensor(X_[:,:,3], -0.5 * mass * length * length * (-X_[:,:,0] * X_[:,:,1] * T.sin(X[:,:,0] - X[:,:,1]) + \
                                                                         gravity / length * T.sin(X[:,:,1])))
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

    RK4 = theano.function([x,h,mass,length,gravity,N],
                          result,
                          allow_input_downcast=True)

    l      = 1.0
    m      = 1.0
    g      = 9.81

    theta1_, theta2_ = np.meshgrid(np.linspace(-np.pi, np.pi, K),
                                   np.linspace(-np.pi, np.pi, K))

    initial_states = np.stack((theta1_, theta2_, np.zeros_like(theta1_), np.zeros_like(theta2_)), axis=2)

    state_array = RK4(initial_states, 0.025, m, l, g, 1000)

    min_crossing_time = []

    for i in range(state_array.shape[1]):
        for j in range(state_array.shape[2]):
            theta_diff    = np.mod(state_array[:,i,j,0] - state_array[:,i,j,1] - np.pi, 2.0 * np.pi)
            crossings     = np.abs(np.diff(theta_diff)) > np.pi

            if np.sum(crossings) == 0:
                min_crossing_time.append(np.nan)
            else:
                min_crossing_time.append(np.min(np.where(crossings)))

    min_crossing_time = np.array(min_crossing_time)

    ax = plt.subplot(111)
    ax.imshow(np.log(min_crossing_time.reshape(K,K)),
              origin='lower',
              extent=[np.min(theta1_), np.max(theta1_), np.min(theta2_), np.max(theta2_)])
    ax.set_aspect('equal')
    plt.savefig(os.path.join(__file__.split('.')[0], 'TimeToFlip.png'), dpi=400)
    plt.show()



def main():
    if 0:
        plot_path()
    else:
        plot_zero_crossing()

if __name__=='__main__':
    main()

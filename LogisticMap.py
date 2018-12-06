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


r = T.fscalar()

N = T.iscalar()
x = T.fscalar()

def step(X):
    X_ = r * X * (1.0 - X)
    return X_

result, _ = theano.scan(fn=step,
                        outputs_info=x,
                        n_steps=N)

logistic_map = theano.function([x,r,N],
                               result,
                               allow_input_downcast=True)





def plot_bifurcation():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_tight_layout(True)

    for r in np.arange(2.0, 4.0, 0.001):
        state_array = logistic_map(0.5, r, 1000)
        ax.scatter(r*np.ones_like(state_array), state_array, c='C0', s=0.0005)

    plt.savefig(os.path.join(__file__.split('.')[0], __file__.split('.')[0]+'.png'), dpi=400)
    plt.show()


def main():
    plot_bifurcation()

if __name__=='__main__':
    main()

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

def step(X):
    return T.square(X) + C

result, _ = theano.scan(fn=step,
                        outputs_info=M,
                        n_steps=50)

f = theano.function([M,C], result, allow_input_downcast=True)

K = 4000

fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_tight_layout(True)
ax.set_aspect('equal')
x,y = np.meshgrid(np.linspace(-2.0, 1.0, K), np.linspace(-1.5, 1.5, K))
mandelbrot = np.sum(np.absolute(f(np.zeros((K,K)), x+y*1j)) < 2.0, axis=0)
ax.contourf(x, y, mandelbrot, cmap='gray')
plt.savefig(os.path.join(__file__.split('.')[0], __file__.split('.')[0]+'.png'), dpi=600)
plt.show()

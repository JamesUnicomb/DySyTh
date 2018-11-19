import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import theano
import theano.tensor as T

sigma = T.fscalar()
rho   = T.fscalar()
beta  = T.fscalar()

h = T.fscalar()
N = T.iscalar()
x = T.fvector()

def f(X):
    X_ = T.zeros_like(X)
    X_ = T.set_subtensor(X_[0], sigma * (X[1] - X[0]))
    X_ = T.set_subtensor(X_[1], X[0] * (rho - X[2]) - X[1])
    X_ = T.set_subtensor(X_[2], X[0] * X[1] - beta * X[2])
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

RK4 = theano.function([x,h,sigma,rho,beta,N],
                      result,
                      allow_input_downcast=True)

test_array = RK4(np.array([1.0, 1.0, 0.0]),
                    0.005, 10.0, 25.0, 8.0 / 3.0, 10000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(test_array[:,0], test_array[:,1], test_array[:,2])
ax.view_init(0.0, -45.0)
plt.show()

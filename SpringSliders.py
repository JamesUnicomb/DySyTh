import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import theano
import theano.tensor as T

kappa = T.fscalar()
rho   = T.fscalar()
beta1 = T.fscalar()
beta2 = T.fscalar()

h = T.fscalar()
N = T.iscalar()
x = T.fvector()

def f(X):
    X_ = T.zeros_like(X)
    X_ = T.set_subtensor(X_[1], (1.0 - T.exp(X[0])) * kappa)
    X_ = T.set_subtensor(X_[2], -T.exp(X[0]) * rho * (beta2 * X[0] + X[2]))
    X_ = T.set_subtensor(X_[0], T.exp(X[0]) * ((beta1 - 1.0) * X[0] + X[1] - X[2]) + X_[1] - X_[2])
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

RK4 = theano.function([x,h,kappa,rho,beta1,beta2,N],
                      result,
                      allow_input_downcast=True)

state_array = RK4(np.array([1.0, 1.0, 0.0]),
                    0.001, 0.852, 0.048, 0.7, 0.84, 10000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(state_array[:,0], state_array[:,1], state_array[:,2])
ax.view_init(0.0, -45.0)
plt.show()

# DySyTh
Dynamical Systems with Theano

## Runge-Kutta - RK4 in Theano
We can solve systems of differential equations using numerical methods. As an example, the equations for the Lorents Attractor:
<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/LorentzEquations.gif" width="150" />
</p>

These written (with more detail in LorentzAttractor.py):
```
def f(X):
    X_ = T.zeros_like(X)
    X_ = T.set_subtensor(X_[0], sigma * (X[1] - X[0]))
    X_ = T.set_subtensor(X_[1], X[0] * (rho - X[2]) - X[1])
    X_ = T.set_subtensor(X_[2], X[0] * X[1] - beta * X[2])
    return X_
```

We can obtain a better estimate numerical estimate with RK4:
```
def step(X):
    k1 = h * f(X)
    k2 = h * f(X + 0.5 * k1)
    k3 = h * f(X + 0.5 * k2)
    k4 = h * f(X + k3)

    X_ = X + (1.0 / 6.0) * k1 + (1.0 / 3.0) * k2 + (1.0 / 3.0) * k3 + (1.0 / 6.0) * k4

    return X_
```

And we can iteratively solve to find the state of the system.

## Lorentz Attractor
<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/LorentzAttractor.gif" width="640" />
</p>


## Double Pendulum
<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/DoublePendulum.gif" width="640" />
</p>

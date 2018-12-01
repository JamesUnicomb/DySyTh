# DySyTh
Dynamical Systems with Theano

## Runge-Kutta - RK4 in Theano
We can solve systems of differential equations using numerical methods. As an example, the equations for the Lorents Attractor:

href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;x'\\&space;y'\\&space;z'&space;\end{bmatrix}=&space;\begin{bmatrix}&space;\sigma&space;(y-x)\\&space;x&space;(\rho&space;-&space;z)&space;-&space;y&space;\\&space;xy&space;-&space;\beta&space;z&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;x'\\&space;y'\\&space;z'&space;\end{bmatrix}=&space;\begin{bmatrix}&space;\sigma&space;(y-x)\\&space;x&space;(\rho&space;-&space;z)&space;-&space;y&space;\\&space;xy&space;-&space;\beta&space;z&space;\end{bmatrix}" title="\begin{bmatrix} x'\\ y'\\ z' \end{bmatrix}= \begin{bmatrix} \sigma (y-x)\\ x (\rho - z) - y \\ xy - \beta z \end{bmatrix}" /></a></p>

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
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/LorenzAttractor/LorenzAttractor.gif" width="640" />
</p>


## Double Pendulum
The equations for the pendulum angles are given by:
<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/DoublePendulum/DoublePendulumEquations1.gif" height="40"/>
</p>

And the momentum:
<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/DoublePendulum/DoublePendulumEquations2.gif" height="40"/>
</p>

We can use Runge-Kutta to solve these equations to produce:
<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/DoublePendulum/DoublePendulum.gif" width="480" />
</p>


### Time for Pendulum to Flip

For each different starting position we time plot how long it takes for the pendulum to flip, producing a fractal pattern.
<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/DoublePendulum/TimeToFlip.png" width="480" />
</p>


## Rabinovich-Fabrikant

The system of equations given by:

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;x'\\&space;y'\\&space;z'&space;\end{bmatrix}=&space;\begin{bmatrix}&space;y(z&space;-&space;1&space;&plus;&space;x^2)&plus;\gamma&space;x\\&space;x(3z&space;&plus;&space;1&space;-&space;x^2)&space;&plus;&space;\gamma&space;y\\&space;-2z(\alpha&space;&plus;&space;xy)&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;x'\\&space;y'\\&space;z'&space;\end{bmatrix}=&space;\begin{bmatrix}&space;y(z&space;-&space;1&space;&plus;&space;x^2)&plus;\gamma&space;x\\&space;x(3z&space;&plus;&space;1&space;-&space;x^2)&space;&plus;&space;\gamma&space;y\\&space;-2z(\alpha&space;&plus;&space;xy)&space;\end{bmatrix}" title="\begin{bmatrix} x'\\ y'\\ z' \end{bmatrix}= \begin{bmatrix} y(z - 1 + x^2)+\gamma x\\ x(3z + 1 - x^2) + \gamma y\\ -2z(\alpha + xy) \end{bmatrix}" /></a></p>


<p align="center">
  <img src="https://github.com/JamesUnicomb/DySyTh/blob/master/RabinovichFabrikant/RabinovichFabrikant.gif" width="480" />
</p>

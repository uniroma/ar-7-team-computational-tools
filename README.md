[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/E033i8DL)
# comptools-assignment2

# Open Issues
* [FIXED] using different initial guess and still obtaining valid estimates.
    
    The issue so far in my code is that the scipy.minimize function always returns the initial guess. This hints at error in the code.
    This could be fixed by imposing bounds on $\sigma^2$, specifically, the bounds have to be set so that $\sigma^2$ is strictly positive.
    This is an unelegant but working solution:
    ```python
    bounds_sigma = tuple((0.000001,np.inf) for _ in range(1))
    ```

* plotting different forecasts.
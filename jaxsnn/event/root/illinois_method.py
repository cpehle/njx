import jax
import jax.numpy as jp

def illinois_method(f, a, b, tol=1e-6, max_iter=600):
    """Find a root of a function in the interval [a, b] using the Illinois method.
    
    The Illinois method is a modification of the secant method that guarantees
    convergence in the presence of a root. It is slower than the bisection method
    but faster than the secant method.
    
    Args:
        f: A function that takes a scalar input and returns a scalar output.
        a: The lower bound of the interval.
        b: The upper bound of the interval.
        tol: The tolerance for the root.
        max_iter: The maximum number of iterations.
        
    Returns:
        The root of the function.
    """

    fa, fb = f(a), f(b)
    
    if jp.sign(fa) == jp.sign(fb):
        raise ValueError("The function must have different signs at the endpoints.")

    def body(val):
        a, b, fa, fb, _ = val
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)

        if jp.abs(fc) < tol:
            return c, a, fa, fb, True
        
        if jp.sign(fc) == jp.sign(fa):
            a = c
            fa = fc
            fb = fb / 2
        else:
            b = c
            fb = fc
        
        return a, b, fa, fb, False


        
    return jax.lax.while_loop(lambda x: not x[-1], body, (a, b, fa, fb, False))

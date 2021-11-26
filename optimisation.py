import numpy as np


class Optimise:

    def __init__ (self):
        pass

    # def gradient_descent (self):
        

    #     while True:
    #         gradient = 
    #         x = x - step_size * direction


    def gradient_descent_1D (self, x0, f, df, epsilon, max_iter:int=5000):
        # compute while dx > epsilon
        
        dx = float('inf')
        x = x0
        step_size = 1

        i = 0

        while (dx > epsilon) and (i < max_iter):
            x_old = x
            x -= step_size * df (x)  # remove gradient

            dx = abs(x - x_old)
            i += 1

        return x

from casadi import *
import numpy as np

class MPC():
    def __init__(self, T, dt, **kwargs):
    '''
    dt: planning timestep
    T: planning horizon
    **kwargs: Model Parameters
    '''
    RC = 6  # 6 hours

    T = 24    # Time horizon
    N = 24*4  # Number of control intervals
    T_l = 20  # Lower comfort bound
    T_h = 24  # Higher comfort bound
    T_i = 22  # Initial temp
    
    x = MX.sym('x')    # States
    to = MX.sym('to')  # Ambient temp
    u = MX.sym('u')    # Controls
    p = vertcat(to,u)

    ode = (to-x)/RC + u/RC

    dae = {'x':x, 'p':p, 'ode':ode}  # x-variable, change in each time step;
                                     # p-parameter, does not change in each time step
    opts = {'tf':T/N}

    F = integrator('F', 'cvodes', dae, opts)

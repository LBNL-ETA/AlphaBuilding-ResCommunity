from casadi import *
import numpy as np

def PWM(signal, cum_error, current_action, dt, limit=0.1):  
    '''
    signal: optimized continuous control action which is the target, a value between 0 and 1
    cum_error: cumulative error, comparing cum_error with limit to change on/off
    current_action: current control action
    dt: control timestep, unit: hour
    limit: the threshold to change on/off
    '''
    pred_error = cum_error + (signal-current_action)*dt
    if pred_error > limit:
        action = 1           # if pred_error > limit, change to on
    elif pred_error <-limit:
        action = 0           # if pred_error < -limit, change to off
    else:
        action = current_action

    return action

class controller_MPC():
    def __init__(self, R, C, T, N, Q_c):
        '''
        R: Thermal Resistance
        C: Thermal Capacity
        T: Time horizon, unit [h]
        N: Number of control steps
        Q_c: maximum cooling capacity, should be negative
        '''
        self.r = R
        self.c = C
        self.t = T
        self.n = N
        self.q_c = Q_c
        self.cum_error = 0
        self.cur_action = 0   # record the current control action, used for PWM

        self.F_x = MX.sym('x')    # States

        self.F_to = MX.sym('to')  # Ambient temp
        self.F_u = MX.sym('u')    # Controls
        self.F_p = vertcat(self.F_to, self.F_u)

        self.F_ode = (self.F_to-self.F_x)/(self.r*self.c) + self.F_u/(self.r*self.c)

        self.F_dae = {'x':self.F_x, 'p':self.F_p, 'ode':self.F_ode}  # x-variable, change in each time step;
                                                                     # p-parameter, does not change in each time step
        self.F_opts = {'tf':self.t/self.n}

        self.F = integrator('F', 'cvodes', self.F_dae, self.F_opts)
    
    def optimize(self, t_current, t_ambient, t_eq, t_lower, t_upper, e_price):
        '''
        return:
            Optimized cooling load for the optimization horizon, unit: degC
        args:
            t_current: Indoor temperature of the current time step
            t_ambient: Ambient temperature of the coming hours, np array of size (N+1,)
            t_eq: Equivalent temperature of the coming hours, np array of size (N+1,)
            t_lower: Lower bound of comfort zone of the coming hours, np array of size (N+1,)
            t_upper: Upper bound of comfort zone of the coming hours, np array of size (N+1,)
            e_price: electricity price of the coming hours, np array of size (N+1,)
        '''
        assert len(t_ambient) == self.n+1, \
            'Number of entries for ambient temperature should equal to control time step plus one'
        assert len(t_eq) == self.n+1, \
            'Number of entries for equivalent temperature of other heat gains should equal to control time step plus one'
        assert len(t_lower) == self.n+1, \
            'Number of entries for lower bound of comfort zone should equal to control time step plus one'
        assert len(t_upper) == self.n+1, \
            'Number of entries for upper bound of comfort zone should equal to control time step plus one'
        assert len(e_price) == self.n+1, \
            'Number of entries for electricity price should equal to control time step plus one'

        opti = casadi.Opti()

        self.opti_X = opti.variable(1,self.n+1)  # Decision variables for states
        self.opti_U = opti.variable(1,self.n)    # Decision variables for controls
        self.opti_P = opti.parameter(4,self.n+1) # Parameter (not optimized over), [ambient temp., t_low, t_up, e_price]

        # opti.minimize(-sum2(self.opti_U))
        opti.minimize(-sum2(self.opti_U*self.opti_P[3,:-1]))

        for k in range(self.n):
            opti.subject_to(self.opti_X[0,k+1]==self.F(x0=self.opti_X[0,k], 
                                                       p =vertcat(self.opti_P[0,k], self.opti_U[0,k]))['xf'])
            opti.subject_to(self.opti_P[1,k+1]<=self.opti_X[0,k+1])
            opti.subject_to(self.opti_X[0,k+1]<=self.opti_P[2,k+1])

        opti.subject_to(self.opti_U<=0)
        opti.subject_to(self.opti_U>=self.q_c*self.r)
        opti.subject_to(self.opti_X[:,0]==t_current)
        
        opti.solver('ipopt')
        
        t_total = t_ambient+t_eq
        opti.set_value(self.opti_P, np.vstack((t_total, t_lower, t_upper, e_price)))
        
        try:
            sol=opti.solve()
            coolingLoad_optimized = sol.value(self.opti_U)
        except:
            coolingLoad_optimized = opti.debug.value(self.opti_U)

        return coolingLoad_optimized

    def action(self, t_current, t_ambient, t_eq, t_lower, t_upper, e_price, type='discrete'):
        '''
        return:
            Control action for this time step, one single value
            If discrete, return a tuple containing the discrete and continuous signal
        args:
            t_current: Indoor temperature of the current time step
            t_ambient: Ambient temperature of the coming hours, np array of size (N+1,)
            t_eq: Equivalent temperature of the coming hours, np array of size (N+1,)
            t_lower: Lower bound of comfort zone of the coming hours, np array of size (N+1,)
            t_upper: Upper bound of comfort zone of the coming hours, np array of size (N+1,)
            e_price: electricity price of the coming hours, np array of size (N+1,)
            type: controller type, if 'discrete', return 0 or 1;
                                   if 'continuous', return the cooling load;            
        '''
        assert type in ['continuous', 'discrete'], \
            'controller type can only be either continuous or discrete'        
        coolingLoad_optimized = self.optimize(t_current, t_ambient, t_eq, t_lower, t_upper, e_price)
        
        if type == 'continuous':
            return coolingLoad_optimized[0]
        elif type == 'discrete':
            signal = coolingLoad_optimized[0]/(self.q_c*self.r)
            dt = self.t/self.n
            self.cur_action = PWM(signal, self.cum_error, self.cur_action, dt, limit=0.1)
            # update the cum_error
            self.cum_error += (signal - self.cur_action)*dt
            return (self.cur_action, signal)